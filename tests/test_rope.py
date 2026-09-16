from __future__ import annotations

import torch

from model import apply_rope, build_rope_cache


def reference_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def test_rope_changes_with_position():
    head_dim = 8
    cos, sin = build_rope_cache(4, head_dim, theta=10000.0, device=torch.device("cpu"), dtype=torch.float32)
    q0 = torch.randn(1, 2, 1, head_dim)
    q1 = q0.clone()
    out0 = apply_rope(q0, cos[:1], sin[:1])
    out1 = apply_rope(q1, cos[1:2], sin[1:2])
    assert not torch.allclose(out0, out1)


def test_apply_rope_matches_reference():
    head_dim = 16
    seq = 8
    cos, sin = build_rope_cache(seq, head_dim, theta=10000.0, device=torch.device("cpu"), dtype=torch.float32)
    x = torch.randn(2, 4, seq, head_dim)
    assert torch.allclose(apply_rope(x, cos, sin), reference_rope(x, cos, sin), atol=1e-6)
