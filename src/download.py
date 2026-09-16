from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "tinystories" / "raw"


def download_tinystories(raw_dir: Path | None = None) -> Path:
    root = raw_dir or RAW_DIR
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        out = root / split
        if out.is_dir() and any(out.iterdir()):
            continue
        ds = load_dataset("roneneldan/TinyStories", split=split)
        ds.save_to_disk(str(out))
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TinyStories to data/tinystories/raw/")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    path = download_tinystories(args.raw_dir)
    print(f"downloaded TinyStories to {path}")


if __name__ == "__main__":
    main()
