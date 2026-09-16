from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from model import LoopedCausalLM
from muon import SingleDeviceMuonWithAuxAdam


@dataclass(frozen=True)
class OptimizerConfig:
    muon_lr: float = 0.02
    adam_lr: float = 3e-4
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.01
    adam_weight_decay: float = 0.01
    adam_betas: tuple[float, float] = (0.9, 0.95)
    ns_steps: int = 5
    use_muon: bool = True


def _is_muon_param(name: str, param: nn.Parameter) -> bool:
    if param.ndim < 2:
        return False
    if "embed" in name:
        return False
    if "norm" in name:
        return False
    if name.endswith(".bias"):
        return False
    return True


def param_groups(model: LoopedCausalLM, cfg: OptimizerConfig) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    muon_params: list[nn.Parameter] = []
    adam_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        if cfg.use_muon and _is_muon_param(name, param):
            muon_params.append(param)
        else:
            adam_params.append(param)
    return muon_params, adam_params


def build_optimizer(model: LoopedCausalLM, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    muon_params, adam_params = param_groups(model, cfg)
    if cfg.use_muon:
        groups: list[dict] = []
        if muon_params:
            groups.append(
                {
                    "params": muon_params,
                    "lr": cfg.muon_lr,
                    "momentum": cfg.muon_momentum,
                    "weight_decay": cfg.muon_weight_decay,
                    "ns_steps": cfg.ns_steps,
                    "use_muon": True,
                }
            )
        if adam_params:
            groups.append(
                {
                    "params": adam_params,
                    "lr": cfg.adam_lr,
                    "betas": cfg.adam_betas,
                    "weight_decay": cfg.adam_weight_decay,
                    "use_muon": False,
                }
            )
        return SingleDeviceMuonWithAuxAdam(groups)

    decay_params = [p for p in muon_params if p.ndim >= 2]
    no_decay_params = adam_params + [p for p in muon_params if p.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.adam_weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.adam_lr,
        betas=cfg.adam_betas,
    )


def optimizer_lrs(optimizer: torch.optim.Optimizer) -> tuple[float | None, float | None]:
    muon_lr: float | None = None
    adam_lr: float | None = None
    for group in optimizer.param_groups:
        if group.get("use_muon"):
            muon_lr = group["lr"]
        else:
            adam_lr = group["lr"]
    if muon_lr is None and len(optimizer.param_groups) == 1:
        adam_lr = optimizer.param_groups[0]["lr"]
    return muon_lr, adam_lr


def set_optimizer_lrs(optimizer: torch.optim.Optimizer, *, muon_lr: float, adam_lr: float) -> None:
    for group in optimizer.param_groups:
        if group.get("use_muon"):
            group["lr"] = muon_lr
        else:
            group["lr"] = adam_lr
