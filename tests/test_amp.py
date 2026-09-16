from __future__ import annotations

from unittest.mock import patch

import torch

from amp import LOSS_DTYPE, resolve_amp, to_loss_dtype


def test_resolve_amp_disabled_on_cpu():
    amp = resolve_amp(torch.device("cpu"), enabled=True)
    assert not amp.enabled


def test_to_loss_dtype():
    x = torch.tensor([1.0], dtype=torch.bfloat16)
    assert to_loss_dtype(x).dtype == LOSS_DTYPE
