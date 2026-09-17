from __future__ import annotations

import torch

from dataset import TokenDataset, fixed_window_starts


def test_batch_shapes_and_label_shift():
    tokens = list(range(200))
    ds = TokenDataset.from_tokens(tokens, seq_len=32, max_samples=10, seed=0)
    x, y = ds[0]
    assert x.shape == (32,)
    assert y.shape == (32,)
    assert torch.equal(x[1:], y[:-1])


def test_fixed_starts_reproducible():
    tokens = list(range(500))
    starts = fixed_window_starts(500 - 33, 5, seed=7)
    ds1 = TokenDataset.from_tokens(tokens, seq_len=32, fixed_starts=starts)
    ds2 = TokenDataset.from_tokens(tokens, seq_len=32, fixed_starts=starts)
    for i in range(len(starts)):
        assert torch.equal(ds1[i][0], ds2[i][0])
        assert torch.equal(ds1[i][1], ds2[i][1])
