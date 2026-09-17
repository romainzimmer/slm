from __future__ import annotations

import pytest
import torch

from amp import AmpConfig
from generate import forward_at_pos, generate_full_forward, generate_greedy
from model import KVCache, LoopedCausalLM, ModelConfig


def _tiny(**kwargs) -> LoopedCausalLM:
    base = dict(
        vocab_size=128,
        max_seq_len=64,
        dim=64,
        num_blocks=1,
        num_heads=2,
        num_kv_heads=2,
        inner_iters=1,
    )
    base.update(kwargs)
    return LoopedCausalLM(ModelConfig.from_preset("tiny", **base))


@pytest.mark.parametrize("inner_iters", [1, 2, 4])
def test_single_token_kv_cache_matches_full(inner_iters: int):
    torch.manual_seed(1)
    model = _tiny(inner_iters=inner_iters)
    model.eval()
    device = torch.device("cpu")
    amp = AmpConfig(enabled=False, dtype=None, scaler=None)
    cache = KVCache(model.cfg.max_seq_len)
    idx = torch.tensor([[7]], dtype=torch.long)
    full = model(idx, inner_iters=inner_iters).logits
    cached = forward_at_pos(model, 7, pos=0, inner_iters=inner_iters, device=device, amp=amp, cache=cache)
    assert torch.allclose(full[:, -1, :], cached, atol=1e-5)


@pytest.mark.parametrize("inner_iters", [1, 2, 4])
def test_prefix_kv_cache_matches_full(inner_iters: int):
    torch.manual_seed(2)
    model = _tiny(inner_iters=inner_iters, num_blocks=2)
    model.eval()
    device = torch.device("cpu")
    amp = AmpConfig(enabled=False, dtype=None, scaler=None)
    prompt = [3, 11, 19, 23, 37]

    idx = torch.tensor([prompt], dtype=torch.long)
    full_logits = model(idx, inner_iters=inner_iters).logits[:, -1, :]

    cache = KVCache(model.cfg.max_seq_len)
    logits = None
    for pos, token_id in enumerate(prompt):
        logits = forward_at_pos(
            model,
            token_id,
            pos=pos,
            inner_iters=inner_iters,
            device=device,
            amp=amp,
            cache=cache,
        )
    assert logits is not None
    assert torch.allclose(full_logits, logits, atol=1e-5)


@pytest.mark.parametrize("inner_iters", [2, 4])
def test_generate_greedy_matches_full_forward(inner_iters: int):
    torch.manual_seed(3)
    model = _tiny(inner_iters=inner_iters, num_blocks=2)
    model.eval()
    device = torch.device("cpu")
    amp = AmpConfig(enabled=False, dtype=None, scaler=None)
    prompt = [1, 2, 3]

    full_ids = generate_full_forward(
        model,
        prompt,
        max_new_tokens=8,
        inner_iters=inner_iters,
        eos_id=-1,
        device=device,
        amp=amp,
    )
    cached_ids = generate_greedy(
        model,
        prompt,
        max_new_tokens=8,
        inner_iters=inner_iters,
        eos_id=-1,
        device=device,
        amp=amp,
    )
    assert cached_ids == full_ids


def test_generate_respects_max_new_tokens():
    model = _tiny(inner_iters=1)
    model.eval()
    amp = AmpConfig(enabled=False, dtype=None, scaler=None)
    ids = generate_full_forward(
        model,
        [1, 2],
        max_new_tokens=5,
        inner_iters=1,
        eos_id=-1,
        device=torch.device("cpu"),
        amp=amp,
    )
    assert len(ids) == 7
