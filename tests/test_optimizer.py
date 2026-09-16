from __future__ import annotations

import torch

from model import LoopedCausalLM, ModelConfig
from optimizer import OptimizerConfig, build_optimizer, param_groups


def test_every_param_in_one_group():
    model = LoopedCausalLM(ModelConfig.from_preset("tiny", vocab_size=128, max_seq_len=64))
    muon_params, adam_params = param_groups(model, OptimizerConfig())
    ids = {id(p) for p in muon_params} | {id(p) for p in adam_params}
    model_ids = {id(p) for p in model.parameters()}
    assert ids == model_ids
    assert not ({id(p) for p in muon_params} & {id(p) for p in adam_params})


def test_adamw_only_mode():
    model = LoopedCausalLM(ModelConfig.from_preset("tiny", vocab_size=128, max_seq_len=64))
    opt = build_optimizer(model, OptimizerConfig(use_muon=False))
    assert opt.__class__.__name__ == "AdamW"
