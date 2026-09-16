from __future__ import annotations

import torch

from dataset import TokenDataset


def test_batch_shapes_and_label_shift():
    tokens = list(range(200))
    ds = TokenDataset.from_tokens(tokens, seq_len=32, max_samples=10, seed=0)
    x, y = ds[0]
    assert x.shape == (32,)
    assert y.shape == (32,)
    assert torch.equal(x[1:], y[:-1])
