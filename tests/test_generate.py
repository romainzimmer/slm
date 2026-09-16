from __future__ import annotations

import torch

from amp import AmpConfig
from generate import forward_at_pos, generate_full_forward
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


def test_single_token_kv_cache_matches_full():
    torch.manual_seed(1)
    model = _tiny(inner_iters=2)
    model.eval()
    device = torch.device("cpu")
    amp = AmpConfig(enabled=False, dtype=None, scaler=None)
    cache = KVCache(model.cfg.max_seq_len)
    idx = torch.tensor([[7]], dtype=torch.long)
    full = model(idx, inner_iters=2).logits
    cached = forward_at_pos(model, 7, pos=0, inner_iters=2, device=device, amp=amp, cache=cache)
    assert torch.allclose(full[:, -1, :], cached, atol=1e-5)


def test_prefix_kv_cache_matches_full_when_single_inner_iter():
    torch.manual_seed(2)
    model = _tiny(inner_iters=1)
    model.eval()
    device = torch.device("cpu")
    amp = AmpConfig(enabled=False, dtype=None, scaler=None)
    prompt = [3, 11, 19, 23]

    idx = torch.tensor([prompt], dtype=torch.long)
    full_logits = model(idx, inner_iters=1).logits[:, -1, :]

    cache = KVCache(model.cfg.max_seq_len)
    logits = None
    for pos, token_id in enumerate(prompt):
        logits = forward_at_pos(
            model,
            token_id,
            pos=pos,
            inner_iters=1,
            device=device,
            amp=amp,
            cache=cache,
        )
    assert logits is not None
    assert torch.allclose(full_logits, logits, atol=1e-5)


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
