from __future__ import annotations

import torch

from model import LoopedCausalLM, ModelConfig, compute_lm_loss


def tiny_model(**kwargs) -> LoopedCausalLM:
    base = dict(
        vocab_size=128,
        max_seq_len=64,
        dim=64,
        num_blocks=1,
        inner_iters=2,
        num_heads=2,
        num_kv_heads=2,
    )
    base.update(kwargs)
    return LoopedCausalLM(ModelConfig.from_preset("tiny", **base))


def test_looped_forward_shape():
    model = tiny_model()
    idx = torch.randint(0, 128, (2, 16))
    out = model(idx)
    assert out.logits.shape == (2, 16, 128)


def test_loss_iters_window():
    model = tiny_model(inner_iters=4, loss_iters=2)
    model.train()
    idx = torch.randint(0, 128, (2, 8))
    out = model(idx)
    assert len(out.iter_logits) == 2


def test_eval_skips_iter_logits():
    model = tiny_model(inner_iters=4, loss_iters=4)
    model.eval()
    out = model(torch.randint(0, 128, (1, 8)))
    assert out.iter_logits == []


def test_eval_collects_supervised_iter_logits():
    model = tiny_model(inner_iters=4, loss_iters=2)
    model.eval()
    out = model(torch.randint(0, 128, (1, 8)), supervised_logits=True, loss_iters=2)
    assert len(out.iter_logits) == 2


def test_weight_tying():
    model = tiny_model(tie_weights=True)
    assert model.lm_head is None
    assert model.get_lm_head_weight() is model.embed.weight


def test_backward_finite_grads():
    model = tiny_model(loss_iters=1)
    model.train()
    idx = torch.randint(0, 128, (2, 8))
    y = torch.randint(0, 128, (2, 8))
    loss = compute_lm_loss(model, idx, y)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
