from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import LoopedCausalLM, ModelConfig
from optimizer import OptimizerConfig, build_optimizer
from train import (
    apply_train_defaults,
    build_model_from_args,
    lr_at_step,
    optimizer_steps_per_epoch,
    resolve_warmup_steps,
    save_checkpoint,
    EpochStats,
    select_sample_prompts,
    should_run_every,
    total_optimizer_steps,
)


def test_optimizer_steps_per_epoch():
    assert optimizer_steps_per_epoch(200, 16) == 13
    assert optimizer_steps_per_epoch(5, 2) == 3
    assert optimizer_steps_per_epoch(4, 2) == 2
    assert total_optimizer_steps(10, 200, 16) == 130


def test_resolve_warmup_steps():
    assert resolve_warmup_steps(warmup_epochs=10, batches_per_epoch=200, grad_accum_steps=16) == 130
    assert resolve_warmup_steps(
        warmup_epochs=10,
        batches_per_epoch=200,
        grad_accum_steps=16,
        warmup_steps=50,
    ) == 50
    assert resolve_warmup_steps(warmup_epochs=0, batches_per_epoch=200, grad_accum_steps=16) == 0


def test_lr_cosine_reaches_floor_at_max_steps():
    base = 1e-3
    warmup = 10
    max_steps = 50
    assert lr_at_step(0, schedule="cosine", warmup_steps=warmup, max_steps=max_steps, base_lr=base, min_lr_ratio=0.0) < base
    assert lr_at_step(warmup, schedule="cosine", warmup_steps=warmup, max_steps=max_steps, base_lr=base, min_lr_ratio=0.0) == base
    mid = lr_at_step((warmup + max_steps) // 2, schedule="cosine", warmup_steps=warmup, max_steps=max_steps, base_lr=base, min_lr_ratio=0.0)
    assert 0.0 < mid < base
    assert lr_at_step(max_steps, schedule="cosine", warmup_steps=warmup, max_steps=max_steps, base_lr=base, min_lr_ratio=0.0) == 0.0


def test_should_run_every():
    assert should_run_every(1, 1)
    assert should_run_every(2, 2)
    assert not should_run_every(1, 2)
    assert should_run_every(4, 2)


def test_select_sample_prompts():
    assert len(select_sample_prompts(3)) == 3
    assert len(select_sample_prompts(0)) == 0


def test_apply_train_defaults():
    args = argparse.Namespace()
    apply_train_defaults(args)
    assert args.viz_samples == 6
    assert args.no_samples is False
    assert args.warmup_epochs == 10.0


def test_checkpoint_logits_roundtrip(tmp_path: Path):
    cfg = ModelConfig.from_preset(
        "tiny",
        vocab_size=128,
        max_seq_len=64,
        seq_len=16,
        dim=64,
        num_blocks=1,
        inner_iters=1,
        num_heads=2,
        num_kv_heads=2,
    )
    model = LoopedCausalLM(cfg)
    opt = build_optimizer(model, OptimizerConfig(use_muon=False))
    args = argparse.Namespace(
        preset="tiny",
        dim=64,
        num_blocks=1,
        inner_iters=1,
        loss_iters=1,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=64,
        seq_len=16,
        vocab_size=128,
        no_weight_tying=False,
        naive_attn=False,
        val_inner_iters=None,
        muon_lr=0.02,
        adam_lr=3e-4,
        no_muon=True,
        no_amp=True,
        seed=0,
        tokenizer_hash="test",
        model="looped-decoder",
    )
    idx = torch.randint(0, 128, (1, 16))
    logits_before = model(idx).logits.detach()
    ckpt_path = tmp_path / "last.pt"
    save_checkpoint(
        ckpt_path,
        model=model,
        optimizer=opt,
        scaler=None,
        epoch=1,
        global_step=1,
        train=EpochStats(0.0, 1.0),
        val=EpochStats(0.0, 1.0),
        args=args,
        best_val_loss=1.0,
    )
    ckpt = torch.load(ckpt_path, weights_only=False)
    model2 = build_model_from_args(args)
    model2.load_state_dict(ckpt["model"])
    logits_after = model2(idx).logits.detach()
    assert torch.allclose(logits_before, logits_after)
