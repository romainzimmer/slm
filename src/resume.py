from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import torch

from train import (
    build_model_from_args,
    load_last_checkpoint,
    require_run_args,
    train_run,
    validate_resume_epochs,
)
from optimizer import OptimizerConfig, build_optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume training from runs/<id>/last.pt")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = parser.parse_args()

    run_dir = cli.run_dir.resolve()
    device = torch.device(cli.device)
    ckpt = load_last_checkpoint(run_dir, device)
    run_args = require_run_args(ckpt, source_name=str(run_dir / "last.pt"))
    completed = int(ckpt["epoch"])
    validate_resume_epochs(completed, cli.epochs)

    args = Namespace(**run_args)
    args.epochs = cli.epochs
    args.device = cli.device

    model = build_model_from_args(args).to(device)
    model.load_state_dict(ckpt["model"])
    opt_cfg = OptimizerConfig(
        muon_lr=args.muon_lr,
        adam_lr=args.adam_lr,
        use_muon=not args.no_muon,
    )
    optimizer = build_optimizer(model, opt_cfg)
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler = None
    if ckpt.get("scaler") is not None:
        from amp import resolve_amp

        amp = resolve_amp(device, enabled=not args.no_amp)
        scaler = amp.scaler
        if scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])

    print(f"resuming {run_dir.name} from epoch {completed + 1}/{cli.epochs}")
    train_run(
        run_dir,
        args,
        start_epoch=completed + 1,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        global_step=int(ckpt.get("global_step", 0)),
        best_val_loss=float(ckpt.get("best_val_loss", float("inf"))),
    )


if __name__ == "__main__":
    main()
