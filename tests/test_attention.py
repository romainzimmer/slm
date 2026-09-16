from __future__ import annotations

import torch

from model import CausalSelfAttention, ModelConfig, build_rope_cache, naive_attention, sdpa_attention


def test_causal_mask_blocks_future():
    q = torch.ones(1, 1, 3, 1)
    k = torch.ones(1, 1, 3, 1)
    v = torch.tensor([[[[10.0], [20.0], [30.0]]]])
    mask = torch.triu(torch.full((3, 3), float("-inf")), diagonal=1)
    out = naive_attention(q, k, v, mask=mask)
    assert out[0, 0, 0, 0] == 10.0
    assert out[0, 0, 1, 0] == 15.0
    assert out[0, 0, 2, 0] == 20.0


def test_naive_matches_sdpa():
    torch.manual_seed(0)
    b, h, t, d = 2, 4, 16, 8
    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t, d)
    v = torch.randn(b, h, t, d)
    mask = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)
    a = naive_attention(q, k, v, mask=mask)
    b_out = sdpa_attention(q, k, v, mask=mask)
    assert torch.allclose(a, b_out, atol=1e-5)


def test_gqa_expands_kv_heads():
    cfg = ModelConfig(
        vocab_size=128,
        dim=64,
        num_blocks=1,
        inner_iters=1,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=32,
    )
    attn = CausalSelfAttention(cfg)
    x = torch.randn(1, 8, cfg.dim)
    head_dim = cfg.dim // cfg.num_heads
    cos, sin = build_rope_cache(8, head_dim, theta=10000.0, device=torch.device("cpu"), dtype=torch.float32)
    out = attn(x, cos=cos, sin=sin, mask=None, naive_attn=False)
    assert out.shape == (1, 8, cfg.dim)
