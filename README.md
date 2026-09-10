# Muon vs AdamW for small MLM encoders

Code for the CSC2005Z project: pretraining Ettin/ModernBERT-style encoders (17M–150M) from scratch under Muon, a tuned AdamW baseline, and a µP-scaled Muon treatment, with singular-value tracking of Muon's momentum buffers.

## Files

| File | What it does |
|---|---|
| `config.py` | Ettin size ladder and `make_config(size, seq_len)` → HF `ModernBertConfig` |
| `prepare_data.py` | Streams FineWeb-Edu, tokenises once with the ModernBERT tokenizer, writes `data/train.bin` and `data/val.bin` (uint16 token streams) |
| `data.py` | `TokenStream` (contiguous 512-token blocks, fixed order) and `mask_tokens` (30% MLM masking, 80/10/10) |
| `muon.py` | Newton–Schulz, the `Muon` optimizer, and `spectral_quantiles` |
| `train.py` | The training loop: parameter groups, three treatments, warmup + cosine schedule, validation, W&B + local logs |
| `sweep.py` | LR sweep, seed and budget phases for one optimizer at one size |
| `run.slurm` | SLURM template for the HPC |
| `kaggle/` | Notebooks for running the 17M grid on Kaggle |

## Setup

```bash
pip install -r requirements.txt
python prepare_data.py --train_tokens 3e9 --val_tokens 2e7
```

3B training tokens covers the 2× AdamW budget at 68M (2 × 20 × 68M = 2.7B). For the 150M stretch use `--train_tokens 6e9`.

## Running

```bash
python train.py --size 17m --treatment muon  --lr 0.02  --seed 0
python train.py --size 17m --treatment adamw --lr 1e-3  --seed 0
python train.py --size 17m --treatment adamw --lr 1e-3  --seed 0 --tokens_per_param 30
python train.py --size 32m --treatment mup   --lr 0.02  --seed 0
```

`--lr` is the swept learning rate: the Muon LR for `muon`/`mup`, the AdamW LR for everything in `adamw`. `--adam_lr` is the (fixed) AdamW LR for the non-body parameters in the Muon treatments. `--tokens_per_param` sets the budget (20 by default; 30 and 40 give the 1.5× and 2× AdamW runs). `--max_steps 100` gives the pilot run.

`--dtype bf16|fp16|fp32` (bf16 needs an Ampere+ GPU; use fp16 on T4/P100, which adds loss scaling; fp32 on CPU/MPS). `--device cuda:1` picks a GPU. A checkpoint is written every `--ckpt_every` steps to `runs/<run>/ckpt.pt`; re-running the same command resumes from it, and a run whose `results.json` exists is skipped.

Set `WANDB_MODE=offline` on nodes without internet and `wandb sync wandb/offline-run-*` afterwards.

### Sweeps

`sweep.py` runs the proposal's protocol for one optimizer at one size, in three phases, skipping finished runs and resuming partial ones. Anything it doesn't recognise is passed through to `train.py`.

```bash
python sweep.py --size 17m --optimizer muon  --phase sweep  --device cuda:0   # 5-point LR sweep, seed 0
python sweep.py --size 17m --optimizer muon  --phase seeds  --device cuda:0   # seeds 1, 2 at the best LR
python sweep.py --size 17m --optimizer adamw --phase budget --device cuda:1   # 1.5x and 2x budgets, 3 seeds, best LR
```

`seeds` and `budget` wait until the sweep for that optimizer has finished, so they can be queued behind it. Sweeps use 5 LRs at 17M/32M and the middle 3 at 68M/150M.

### Kaggle

`kaggle/` has two notebooks: `muon-encoder-data.ipynb` (CPU, tokenises 700M tokens once) and `muon-encoder-train-17m.ipynb` (2× T4, runs the whole 17M grid, one worker per GPU, stops itself before the 12 h limit and resumes from its own previous output on the next version).

## What each treatment does

Parameters are split into two groups everywhere: **body** = every 2D weight inside `model.layers.*` (Wqkv, Wo, mlp.Wi, mlp.Wo), which gets weight decay; **rest** = embeddings, MLM head, norms, decoder bias, which gets none.

- `muon`: Muon on body, AdamW on rest.
- `adamw`: AdamW on both groups, same decay assignment. Only the body update rule differs from `muon`.
- `mup`: identical to `muon` but with `lr · sqrt(256/d)` and `wd · 256/d` on the body, where `d` is the model width. At 17M it is exactly `muon`.

Data order is fixed by the token stream and identical across treatments and seeds; the seed changes initialisation and the masking pattern.

## Spectral tracking

Every `--spectral_every` steps, the Muon step computes the singular values of the matrix Newton–Schulz actually receives, `A = G / ‖G‖_F` (with Nesterov, `G = (1−β)·grad + β·M`), and records for each body matrix:

- `sv_q{0.1,0.25,0.5,0.75,0.9}`: the singular value at rank ⌈q·r⌉ in descending order (q = 0.1 is a large one, q = 0.9 a small one), following Magakyan et al.
- `frac_below_threshold`: the fraction of singular values that 5-step Newton–Schulz cannot orthogonalise (maps to < 0.1). The threshold is computed from the polynomial coefficients in `muon.py`: ≈ 2×10⁻⁴ for Keller Jordan's coefficients (3.4445, −4.775, 2.0315), and ≈ 3×10⁻³ for the textbook (2, −1.5, 0.5) coefficients that Magakyan et al.'s 0.003 figure refers to.

Everything is logged to W&B under `spectra/<layer>/<metric>` and to `runs/<run>/spectra.jsonl` (one line per layer per logged step) for offline power-law fitting. Final validation loss and run metadata go to `runs/<run>/results.json`.
