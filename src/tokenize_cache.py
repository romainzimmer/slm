from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import BinaryIO

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

from dataset import DATA_DIR
from tokenizer import DEFAULT_TOKENIZER_DIR, TextTokenizer

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "tinystories" / "raw"
WRITE_BUFFER_TOKENS = 1_000_000


def tokenize_split(
    split: str,
    *,
    raw_dir: Path,
    tokenizer: TextTokenizer,
    output_dir: Path,
) -> int:
    split_dir = raw_dir / ("train" if split == "train" else "validation")
    if not split_dir.is_dir():
        raise FileNotFoundError(f"{split_dir} not found")
    ds = load_from_disk(str(split_dir))
    out_name = "train.bin" if split == "train" else "val.bin"
    out_path = output_dir / out_name
    total = 0
    buffer: list[int] = []

    def flush(out: BinaryIO) -> None:
        nonlocal total
        if not buffer:
            return
        np.asarray(buffer, dtype=np.uint16).tofile(out)
        total += len(buffer)
        buffer.clear()

    with out_path.open("wb") as out:
        for row in tqdm(ds, desc=f"tokenize {split}"):
            buffer.extend(tokenizer.encode(row["text"]))
            buffer.append(tokenizer.eos_id)
            if len(buffer) >= WRITE_BUFFER_TOKENS:
                flush(out)
        flush(out)
    return total


def write_meta(
    output_dir: Path,
    *,
    tokenizer: TextTokenizer,
    train_tokens: int,
    val_tokens: int,
) -> dict:
    meta = {
        "tokenizer_hash": tokenizer.meta()["tokenizer_hash"],
        "vocab_size": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_id,
        "dtype": "uint16",
        "splits": {
            "train": {"tokens": train_tokens, "file": "train.bin"},
            "validation": {"tokens": val_tokens, "file": "val.bin"},
        },
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode TinyStories into memmap .bin shards")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = TextTokenizer.load(args.tokenizer_dir)
    train_tokens = tokenize_split("train", raw_dir=args.raw_dir, tokenizer=tokenizer, output_dir=args.output_dir)
    val_tokens = tokenize_split("validation", raw_dir=args.raw_dir, tokenizer=tokenizer, output_dir=args.output_dir)
    meta = write_meta(args.output_dir, tokenizer=tokenizer, train_tokens=train_tokens, val_tokens=val_tokens)
    print(f"wrote {args.output_dir}/train.bin ({train_tokens} tokens), val.bin ({val_tokens} tokens)")
    print(f"tokenizer_hash={meta['tokenizer_hash']}")


if __name__ == "__main__":
    main()
