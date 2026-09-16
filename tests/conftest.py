from __future__ import annotations

import pytest
import torch

from model import LoopedCausalLM, ModelConfig


@pytest.fixture
def tiny_cfg() -> ModelConfig:
    return ModelConfig.from_preset(
        "tiny",
        vocab_size=128,
        max_seq_len=64,
        dim=64,
        num_blocks=1,
        inner_iters=2,
        num_heads=2,
        num_kv_heads=2,
        loss_iters=1,
    )


@pytest.fixture
def tiny_model(tiny_cfg: ModelConfig) -> LoopedCausalLM:
    torch.manual_seed(0)
    return LoopedCausalLM(tiny_cfg)
