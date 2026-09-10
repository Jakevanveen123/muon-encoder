import argparse
import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--train_tokens", type=float, default=3e9)
parser.add_argument("--val_tokens", type=float, default=2e7)
parser.add_argument("--out_dir", default="data")
parser.add_argument("--docs_per_batch", type=int, default=1000)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
tokenizer.model_max_length = int(1e9)
dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
)
docs = iter(dataset)


def write_split(name, budget):
    path = os.path.join(args.out_dir, f"{name}.bin")
    written = 0
    with open(path, "wb") as f:
        while written < budget:
            texts = [next(docs)["text"] for _ in range(args.docs_per_batch)]
            ids = tokenizer(texts)["input_ids"]
            flat = np.concatenate([np.array(x, dtype=np.uint16) for x in ids])
            f.write(flat.tobytes())
            written += len(flat)
            print(f"\r{name}: {written / 1e6:.1f}M tokens", end="", flush=True)
    print()


write_split("val", args.val_tokens)
write_split("train", args.train_tokens)
