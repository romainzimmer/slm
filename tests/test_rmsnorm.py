from __future__ import annotations

import torch

from model import RMSNorm


def reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x * rms * weight


def test_rmsnorm_matches_reference():
    norm = RMSNorm(32)
    x = torch.randn(2, 10, 32)
    expected = reference_rmsnorm(x, norm.weight, norm.eps)
    assert torch.allclose(norm(x), expected, atol=1e-6)
