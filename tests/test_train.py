from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import LoopedCausalLM, ModelConfig
from optimizer import OptimizerConfig, build_optimizer
from train import build_model_from_args, save_checkpoint, EpochStats


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
        no_input_injection=False,
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
