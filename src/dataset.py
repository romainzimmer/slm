from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "tinystories"


def load_meta(data_dir: Path | None = None) -> dict:
    root = data_dir or DATA_DIR
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path} not found; run tokenize-cache first")
    return json.loads(meta_path.read_text())


def require_token_bins(data_dir: Path | None = None) -> tuple[Path, Path, dict]:
    root = data_dir or DATA_DIR
    meta = load_meta(root)
    train_path = root / "train.bin"
    val_path = root / "val.bin"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(
            f"missing {train_path.name} or {val_path.name}; run tokenize-cache first"
        )
    return train_path, val_path, meta


class TokenDataset(Dataset):
    def __init__(
        self,
        bin_path: Path,
        *,
        seq_len: int,
        max_samples: int | None = None,
        seed: int = 0,
    ):
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.window = seq_len + 1
        if len(self.data) < self.window:
            raise ValueError(f"{bin_path} too short for seq_len={seq_len}")
        self.max_starts = len(self.data) - self.window
        self.num_samples = self.max_starts if max_samples is None else min(max_samples, self.max_starts)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        del index
        start = int(self.rng.integers(0, self.max_starts))
        chunk = np.array(self.data[start : start + self.window], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

    @classmethod
    def from_tokens(
        cls,
        tokens: list[int],
        *,
        seq_len: int,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> TokenDataset:
        tmp = cls.__new__(cls)
        tmp.bin_path = Path("memory")
        tmp.seq_len = seq_len
        tmp.data = np.array(tokens, dtype=np.uint16)
        tmp.window = seq_len + 1
        tmp.max_starts = len(tmp.data) - tmp.window
        tmp.num_samples = tmp.max_starts if max_samples is None else min(max_samples, tmp.max_starts)
        tmp.rng = np.random.default_rng(seed)
        return tmp
