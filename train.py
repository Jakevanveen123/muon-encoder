import argparse
import json
import math
import os
import sys
import time
import torch
import wandb
from transformers import ModernBertForMaskedLM
from config import make_config, BASE_WIDTH
from data import TokenStream, mask_tokens
from muon import Muon

parser = argparse.ArgumentParser()
parser.add_argument("--size", default="17m")
parser.add_argument("--treatment", default="muon", choices=["muon", "adamw", "mup"])
parser.add_argument("--lr", type=float, default=0.02)
parser.add_argument("--adam_lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.95)
parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
parser.add_argument("--tokens_per_param", type=float, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--mask_rate", type=float, default=0.3)
parser.add_argument("--warmup_frac", type=float, default=0.05)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eval_every", type=int, default=500)
parser.add_argument("--eval_batches", type=int, default=20)
parser.add_argument("--final_eval_batches", type=int, default=200)
parser.add_argument("--spectral_every", type=int, default=20)
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument("--ckpt_every", type=int, default=500)
parser.add_argument("--max_steps", type=int, default=None)
parser.add_argument("--data_dir", default="data")
parser.add_argument("--out_dir", default="runs")
parser.add_argument("--wandb_project", default="muon-encoder")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
parser.add_argument("--compile", action="store_true")
args = parser.parse_args()

run_name = (
    f"{args.size}-{args.treatment}-lr{args.lr}-wd{args.wd}"
    f"-tpp{args.tokens_per_param:g}-s{args.seed}"
)
run_dir = os.path.join(args.out_dir, run_name)
results_path = os.path.join(run_dir, "results.json")
ckpt_path = os.path.join(run_dir, "ckpt.pt")
if os.path.exists(results_path):
    print(f"{run_name} already finished")
    sys.exit()
os.makedirs(run_dir, exist_ok=True)

torch.manual_seed(args.seed)
device = args.device
dtype = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[args.dtype]
autocast = torch.autocast(device.split(":")[0], dtype=dtype, enabled=args.dtype != "fp32")
scaler = torch.amp.GradScaler(enabled=args.dtype == "fp16")

config = make_config(args.size, args.seq_len)
model = ModernBertForMaskedLM(config).to(device)
raw_model = model
param_names = {p: n for n, p in model.named_parameters()}
n_params = sum(p.numel() for p in model.parameters())
n_body = sum(p.numel() for n, p in model.named_parameters() if p.ndim == 2 and "layers." in n)


def build_optimizers(model):
    body = [p for n, p in model.named_parameters() if p.ndim == 2 and "layers." in n]
    rest = [p for n, p in model.named_parameters() if not (p.ndim == 2 and "layers." in n)]
    lr, wd = args.lr, args.wd
    if args.treatment == "mup":
        width_ratio = BASE_WIDTH / config.hidden_size
        lr, wd = lr * math.sqrt(width_ratio), wd * width_ratio
    if args.treatment == "adamw":
        groups = [dict(params=body, weight_decay=wd), dict(params=rest, weight_decay=0.0)]
        return None, torch.optim.AdamW(groups, lr=lr, betas=args.betas)
    ns_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    muon = Muon(body, lr=lr, momentum=args.momentum, weight_decay=wd, ns_dtype=ns_dtype)
    adam = torch.optim.AdamW(rest, lr=args.adam_lr, betas=args.betas, weight_decay=0.0)
    return muon, adam


muon, adam = build_optimizers(model)
optimizers = [opt for opt in (muon, adam) if opt is not None]
for opt in optimizers:
    for group in opt.param_groups:
        group["base_lr"] = group["lr"]

tokens_per_step = args.batch_size * args.seq_len
total_steps = math.ceil(args.tokens_per_param * n_params / tokens_per_step)
if args.max_steps is not None:
    total_steps = min(total_steps, args.max_steps)
warmup_steps = int(args.warmup_frac * total_steps)


def lr_multiplier(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


train_stream = TokenStream(os.path.join(args.data_dir, "train.bin"), args.seq_len, args.batch_size)
val_stream = TokenStream(os.path.join(args.data_dir, "val.bin"), args.seq_len, args.batch_size)
mask_gen = torch.Generator().manual_seed(args.seed)

start_step = 0
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    for opt, state in zip(optimizers, ckpt["optimizers"]):
        opt.load_state_dict(state)
    scaler.load_state_dict(ckpt["scaler"])
    train_stream.pos = ckpt["stream_pos"]
    mask_gen.set_state(ckpt["mask_gen"].cpu())
    start_step = ckpt["step"] + 1
    print(f"resuming {run_name} from step {start_step}")

if args.compile:
    model = torch.compile(model)


def save_checkpoint(step):
    torch.save(
        dict(
            model=raw_model.state_dict(),
            optimizers=[opt.state_dict() for opt in optimizers],
            scaler=scaler.state_dict(),
            stream_pos=train_stream.pos,
            mask_gen=mask_gen.get_state(),
            step=step,
        ),
        ckpt_path,
    )


@torch.no_grad()
def evaluate(n_batches):
    model.eval()
    val_stream.pos = 0
    gen = torch.Generator().manual_seed(1234)
    losses = []
    for _ in range(n_batches):
        inputs, labels = mask_tokens(val_stream.next_batch(), args.mask_rate, gen)
        with autocast:
            loss = model(input_ids=inputs.to(device), labels=labels.to(device)).loss
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def log_spectra(step, spectra_file):
    for p, quantiles in muon.spectra.items():
        layer = param_names[p].replace("model.layers.", "").replace(".weight", "")
        spectra_file.write(json.dumps(dict(step=step, layer=layer, **quantiles)) + "\n")
        wandb.log({f"spectra/{layer}/{k}": v for k, v in quantiles.items()}, step=step)


wandb.init(
    project=args.wandb_project,
    name=run_name,
    id=run_name.replace(".", "p"),
    resume="allow",
    config=vars(args),
)
wandb.config.update(dict(n_params=n_params, n_body_params=n_body, total_steps=total_steps))
print(f"{run_name}: {n_params / 1e6:.1f}M params ({n_body / 1e6:.1f}M body), {total_steps} steps")
spectra_path = os.path.join(run_dir, "spectra.jsonl")
if start_step > 0 and os.path.exists(spectra_path):
    kept = [line for line in open(spectra_path) if json.loads(line)["step"] < start_step]
    open(spectra_path, "w").writelines(kept)
spectra_file = open(spectra_path, "a")
model.train()
start = time.time()

for step in range(start_step, total_steps):
    mult = lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * mult

    inputs, labels = mask_tokens(train_stream.next_batch(), args.mask_rate, mask_gen)
    with autocast:
        loss = model(input_ids=inputs.to(device), labels=labels.to(device)).loss
    scaler.scale(loss).backward()
    for opt in optimizers:
        scaler.unscale_(opt)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    track = muon is not None and step % args.spectral_every == 0
    if muon is not None:
        scaler.step(muon, track_spectra=track)
    scaler.step(adam)
    scaler.update()
    model.zero_grad(set_to_none=True)

    if track:
        log_spectra(step, spectra_file)
    if step % args.log_every == 0:
        elapsed = time.time() - start
        tokens_per_sec = (step + 1 - start_step) * tokens_per_step / elapsed
        wandb.log(
            dict(
                train_loss=loss.item(),
                grad_norm=grad_norm.item(),
                lr_mult=mult,
                tokens=(step + 1) * tokens_per_step,
                tokens_per_sec=tokens_per_sec,
            ),
            step=step,
        )
        print(f"step {step}/{total_steps} loss {loss.item():.4f} {tokens_per_sec:,.0f} tok/s")
    if step % args.eval_every == 0 and step > 0:
        wandb.log(dict(val_loss=evaluate(args.eval_batches)), step=step)
    if step % args.ckpt_every == 0 and step > 0:
        save_checkpoint(step)

final_val_loss = evaluate(args.final_eval_batches)
wandb.log(dict(val_loss=final_val_loss, final_val_loss=final_val_loss), step=total_steps)
spectra_file.close()
results = dict(
    run_name=run_name,
    final_val_loss=final_val_loss,
    n_params=n_params,
    n_body_params=n_body,
    tokens=total_steps * tokens_per_step,
    steps=total_steps,
    wall_time=time.time() - start,
    args=vars(args),
)
json.dump(results, open(results_path, "w"), indent=2)
print(f"final val loss {final_val_loss:.4f}")
wandb.finish()
