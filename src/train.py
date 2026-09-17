from __future__ import annotations

import argparse
import json
import math
import random
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from amp import AmpConfig, autocast_context, resolve_amp
from dataset import TokenDataset, fixed_window_starts, load_meta, require_token_bins
from model import LoopedCausalLM, ModelConfig, PRESETS, compute_lm_loss
from optimizer import OptimizerConfig, build_optimizer, optimizer_lrs, set_optimizer_lrs
from tokenizer import TextTokenizer

DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def make_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(4)}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def should_run_every(epoch: int, every: int) -> bool:
    if every <= 1:
        return True
    return epoch % every == 0


def apply_train_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "val_every": 1,
        "save_every": 1,
        "no_samples": False,
        "sample_every": 1,
        "viz_samples": 6,
        "viz_max_new_tokens": 64,
        "viz_batch_size": 4,
        "no_val_bpc": False,
        "val_loss_iters": None,
        "warmup_epochs": 10.0,
        "warmup_steps": None,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def select_sample_prompts(viz_samples: int) -> list[str]:
    from generate import DEFAULT_PROMPTS

    n = max(0, min(viz_samples, len(DEFAULT_PROMPTS)))
    return DEFAULT_PROMPTS[:n]


LR_SCHEDULES = ("cosine", "linear", "constant")


def optimizer_steps_per_epoch(batches_per_epoch: int, grad_accum_steps: int) -> int:
    if batches_per_epoch < 1:
        raise ValueError("batches_per_epoch must be >= 1")
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    return (batches_per_epoch + grad_accum_steps - 1) // grad_accum_steps


def total_optimizer_steps(epochs: int, batches_per_epoch: int, grad_accum_steps: int) -> int:
    return epochs * optimizer_steps_per_epoch(batches_per_epoch, grad_accum_steps)


def resolve_warmup_steps(
    *,
    warmup_epochs: float,
    batches_per_epoch: int,
    grad_accum_steps: int,
    warmup_steps: int | None = None,
) -> int:
    if warmup_steps is not None:
        return warmup_steps
    if warmup_epochs <= 0:
        return 0
    steps_per_epoch = optimizer_steps_per_epoch(batches_per_epoch, grad_accum_steps)
    return max(1, round(warmup_epochs * steps_per_epoch))


def lr_at_step(
    step: int,
    *,
    schedule: str,
    warmup_steps: int,
    max_steps: int,
    base_lr: float,
    min_lr_ratio: float,
) -> float:
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"unknown lr schedule {schedule!r}; choose from {LR_SCHEDULES}")
    min_lr = base_lr * min_lr_ratio
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    if schedule == "constant":
        return base_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    if schedule == "linear":
        return base_lr + (min_lr - base_lr) * progress
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass
class EpochStats:
    loss: float
    ppl: float
    bpc: float | None = None
    tokens_per_sec: float | None = None
    step_time_ms: float | None = None
    grad_norm: float | None = None
    mem_allocated_mb: float | None = None


@dataclass
class SplitMetricsAccumulator:
    total_loss: torch.Tensor
    total_tokens: torch.Tensor

    @classmethod
    def empty(cls, device: torch.device) -> SplitMetricsAccumulator:
        return cls(
            total_loss=torch.zeros((), device=device),
            total_tokens=torch.zeros((), device=device, dtype=torch.long),
        )

    def add_batch(self, loss: torch.Tensor, n: int) -> None:
        self.total_loss += loss.detach() * n
        self.total_tokens += n

    def avg_loss(self) -> float:
        tokens = int(self.total_tokens.item())
        if tokens == 0:
            return 0.0
        return (self.total_loss / tokens).item()


def compute_bpc(
    loss_sum: float,
    target_tokens: torch.Tensor,
    tokenizer: TextTokenizer,
) -> float:
    text = tokenizer.decode(target_tokens.reshape(-1).tolist())
    n_chars = len(text.encode("utf-8"))
    if n_chars == 0:
        return float("nan")
    total_bits = loss_sum * math.log(2)
    return total_bits / n_chars


def build_model_from_args(args: argparse.Namespace) -> LoopedCausalLM:
    overrides: dict = {
        "seq_len": getattr(args, "seq_len", 512),
        "vocab_size": getattr(args, "vocab_size", 8192),
        "tie_weights": not getattr(args, "no_weight_tying", False),
        "use_naive_attn": getattr(args, "naive_attn", False),
    }
    for key, arg_name in (
        ("dim", "dim"),
        ("num_blocks", "num_blocks"),
        ("inner_iters", "inner_iters"),
        ("num_heads", "num_heads"),
        ("num_kv_heads", "num_kv_heads"),
        ("max_seq_len", "max_seq_len"),
        ("loss_iters", "loss_iters"),
    ):
        val = getattr(args, arg_name.replace("-", "_"), None)
        if val is not None:
            overrides[key] = val
    preset = getattr(args, "preset", None)
    if preset:
        cfg = ModelConfig.from_preset(preset, **overrides)
    else:
        cfg = ModelConfig(**overrides)
    return LoopedCausalLM(cfg)


def save_run_config(run_dir: Path, args: argparse.Namespace) -> None:
    history = {"run_id": run_dir.name, "args": json_safe(vars(args)), "epochs": []}
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))


def save_epoch_metrics(
    run_dir: Path,
    *,
    epoch: int,
    train: EpochStats,
    val: EpochStats,
    args: argparse.Namespace,
    lr_muon: float | None,
    lr_adam: float | None,
) -> None:
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text())
    history["args"] = json_safe(vars(args))
    history["epochs"] = [e for e in history["epochs"] if e["epoch"] != epoch]
    row = {
        "epoch": epoch,
        "train_loss": train.loss,
        "val_loss": val.loss,
        "train_ppl": train.ppl,
        "val_ppl": val.ppl,
        "train_bpc": train.bpc,
        "val_bpc": val.bpc,
        "lr_muon": lr_muon,
        "lr_adam": lr_adam,
        "tokens_per_sec": train.tokens_per_sec,
        "step_time_ms": train.step_time_ms,
        "grad_norm": train.grad_norm,
        "mem_allocated_mb": val.mem_allocated_mb,
    }
    history["epochs"].append(row)
    history["epochs"].sort(key=lambda e: e["epoch"])
    history_path.write_text(json.dumps(history, indent=2))


def save_checkpoint(
    path: Path,
    *,
    model: LoopedCausalLM,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    global_step: int,
    train: EpochStats,
    val: EpochStats,
    args: argparse.Namespace,
    best_val_loss: float,
) -> None:
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "train_loss": train.loss,
        "val_loss": val.loss,
        "best_val_loss": best_val_loss,
        "args": json_safe(vars(args)),
    }
    torch.save(payload, path)


def load_last_checkpoint(run_dir: Path, device: torch.device) -> dict:
    path = run_dir / "last.pt"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    return torch.load(path, map_location=device, weights_only=False)


def validate_resume_epochs(completed_epoch: int, new_epochs: int) -> None:
    if new_epochs <= completed_epoch:
        raise ValueError(f"--epochs {new_epochs} must exceed completed epoch {completed_epoch}")


def resolve_checkpoint_target(path: Path) -> tuple[Path, Path]:
    path = path.resolve()
    if path.is_dir():
        checkpoint = path / "best.pt"
        if not checkpoint.is_file():
            checkpoint = path / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"no checkpoint in {path}")
        return path, checkpoint
    if path.suffix != ".pt":
        raise ValueError(f"expected run directory or .pt checkpoint, got {path}")
    if path.parent.name == "epochs":
        return path.parent.parent, path
    return path.parent, path


def require_run_args(source: dict, *, source_name: str) -> dict:
    args = source.get("args", source)
    required = ("vocab_size", "dim", "num_blocks", "inner_iters", "seq_len")
    missing = [k for k in required if k not in args]
    if missing:
        raise ValueError(f"{source_name} missing keys: {', '.join(missing)}")
    return args


@torch.inference_mode()
def measure_split(
    model: LoopedCausalLM,
    loader: DataLoader,
    device: torch.device,
    *,
    inner_iters: int | None,
    loss_iters: int | None,
    amp: AmpConfig,
    tokenizer: TextTokenizer | None = None,
    max_batches: int | None = None,
    use_cuda: bool = False,
    include_bpc: bool = True,
    epoch: int = 1,
    epochs: int = 1,
) -> EpochStats:
    model.eval()
    acc = SplitMetricsAccumulator.empty(device)
    bpc_targets: torch.Tensor | None = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    progress = tqdm(
        loader,
        desc=f"epoch {epoch}/{epochs} val",
        leave=False,
        mininterval=0.5,
    )
    for batch_idx, (x, y) in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=use_cuda)
        y = y.to(device, non_blocking=use_cuda)
        with autocast_context(device, amp):
            loss = compute_lm_loss(
                model,
                x,
                y,
                inner_iters=inner_iters,
                loss_iters=loss_iters,
                supervised_logits=True,
            )
        n = y.numel()
        acc.add_batch(loss, n)
        if include_bpc and bpc_targets is None:
            bpc_targets = y.detach()
        progress.set_postfix(loss=f"{acc.avg_loss():.4f}", refresh=False)
    progress.close()

    avg = acc.avg_loss()
    bpc = None
    if include_bpc and tokenizer is not None and bpc_targets is not None:
        tokens = int(acc.total_tokens.item())
        bpc = compute_bpc(avg * tokens, bpc_targets, tokenizer)
    mem_mb = None
    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return EpochStats(loss=avg, ppl=math.exp(min(avg, 20)), bpc=bpc, mem_allocated_mb=mem_mb)


def train_epoch(
    model: LoopedCausalLM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp: AmpConfig,
    grad_accum_steps: int,
    grad_clip: float,
    global_step: int,
    warmup_steps: int,
    max_steps: int,
    lr_schedule: str,
    min_lr_ratio: float,
    muon_base_lr: float,
    adam_base_lr: float,
    inner_iters: int | None,
    batches_per_epoch: int,
    epoch: int,
    epochs: int,
    use_cuda: bool = False,
) -> tuple[EpochStats, int]:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    grad_norm_val: float | None = None
    import time

    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loader_iter = iter(loader)
    accum_count = 0
    progress = tqdm(range(batches_per_epoch), desc=f"epoch {epoch}/{epochs} train", leave=False)
    for batch_idx in progress:
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)
        x = x.to(device, non_blocking=use_cuda)
        y = y.to(device, non_blocking=use_cuda)
        lr_kw = dict(
            schedule=lr_schedule,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
        )
        lr_muon = lr_at_step(global_step, base_lr=muon_base_lr, **lr_kw)
        lr_adam = lr_at_step(global_step, base_lr=adam_base_lr, **lr_kw)
        set_optimizer_lrs(optimizer, muon_lr=lr_muon, adam_lr=lr_adam)

        with autocast_context(device, amp):
            loss = compute_lm_loss(model, x, y, inner_iters=inner_iters)
            loss = loss / grad_accum_steps
        if amp.scaler is not None:
            amp.scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_count += 1
        if accum_count >= grad_accum_steps:
            if amp.scaler is not None:
                amp.scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norm_val = float(grad_norm)
            if amp.scaler is not None:
                amp.scaler.step(optimizer)
                amp.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            accum_count = 0

        n = y.numel()
        total_loss += loss.item() * grad_accum_steps * n
        total_tokens += n

    if accum_count > 0:
        if amp.scaler is not None:
            amp.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if amp.scaler is not None:
            amp.scaler.step(optimizer)
            amp.scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    elapsed = time.perf_counter() - t0
    avg = total_loss / max(1, total_tokens)
    tps = total_tokens / max(elapsed, 1e-6)
    return (
        EpochStats(
            loss=avg,
            ppl=math.exp(min(avg, 20)),
            tokens_per_sec=tps,
            step_time_ms=(elapsed * 1000) / max(batches_per_epoch, 1),
            grad_norm=grad_norm_val,
        ),
        global_step,
    )


def train_run(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    start_epoch: int = 1,
    model: LoopedCausalLM | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
    global_step: int = 0,
    best_val_loss: float = float("inf"),
) -> None:
    apply_train_defaults(args)
    device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    require_token_bins()
    tokenizer = TextTokenizer.load()

    if model is None:
        model = build_model_from_args(args).to(device)
    sync_args_from_model(args, model)
    if optimizer is None:
        opt_cfg = OptimizerConfig(
            muon_lr=args.muon_lr,
            adam_lr=args.adam_lr,
            use_muon=not args.no_muon,
        )
        optimizer = build_optimizer(model, opt_cfg)

    amp = resolve_amp(device, enabled=not args.no_amp)
    if scaler is None and amp.scaler is not None:
        scaler = amp.scaler

    train_path, val_path, _ = require_token_bins()
    train_ds = TokenDataset(
        train_path,
        seq_len=args.seq_len,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    val_probe = TokenDataset(val_path, seq_len=args.seq_len, max_samples=1, seed=args.seed + 1)
    val_starts = fixed_window_starts(val_probe.max_starts, args.val_max_samples, args.seed + 1)
    val_ds = TokenDataset(
        val_path,
        seq_len=args.seq_len,
        seed=args.seed + 1,
        fixed_starts=val_starts,
    )
    loader_kw: dict = {
        "shuffle": False,
        "num_workers": args.num_workers,
    }
    if use_cuda and args.num_workers == 0:
        loader_kw["pin_memory"] = True
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, **loader_kw)

    val_inner = args.val_inner_iters or model.cfg.inner_iters
    val_loss_iters = args.val_loss_iters or model.cfg.loss_iters
    max_steps = args.max_steps or total_optimizer_steps(
        args.epochs,
        args.batches_per_epoch,
        args.grad_accum_steps,
    )
    warmup_steps = resolve_warmup_steps(
        warmup_epochs=args.warmup_epochs,
        batches_per_epoch=args.batches_per_epoch,
        grad_accum_steps=args.grad_accum_steps,
        warmup_steps=args.warmup_steps,
    )
    args.warmup_steps = warmup_steps
    sample_prompts = select_sample_prompts(args.viz_samples)
    last_val_stats = EpochStats(loss=float("nan"), ppl=float("nan"))

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            amp=amp,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip=args.grad_clip,
            global_step=global_step,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            lr_schedule=args.lr_schedule,
            min_lr_ratio=args.min_lr_ratio,
            muon_base_lr=args.muon_lr,
            adam_base_lr=args.adam_lr,
            inner_iters=model.cfg.inner_iters,
            batches_per_epoch=args.batches_per_epoch,
            epoch=epoch,
            epochs=args.epochs,
            use_cuda=use_cuda,
        )

        if should_run_every(epoch, args.val_every):
            val_stats = measure_split(
                model,
                val_loader,
                device,
                inner_iters=val_inner,
                loss_iters=val_loss_iters,
                amp=amp,
                tokenizer=tokenizer if not args.no_val_bpc else None,
                max_batches=args.val_batches,
                use_cuda=use_cuda,
                include_bpc=not args.no_val_bpc,
                epoch=epoch,
                epochs=args.epochs,
            )
            last_val_stats = val_stats
        else:
            val_stats = last_val_stats

        lr_muon, lr_adam = optimizer_lrs(optimizer)
        save_epoch_metrics(
            run_dir,
            epoch=epoch,
            train=train_stats,
            val=val_stats,
            args=args,
            lr_muon=lr_muon,
            lr_adam=lr_adam,
        )

        if should_run_every(epoch, args.save_every):
            save_checkpoint(
                run_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                train=train_stats,
                val=val_stats,
                args=args,
                best_val_loss=best_val_loss,
            )
        if not math.isnan(val_stats.loss) and val_stats.loss < best_val_loss:
            best_val_loss = val_stats.loss
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                train=train_stats,
                val=val_stats,
                args=args,
                best_val_loss=best_val_loss,
            )
        if args.save_epochs:
            epoch_dir = run_dir / "epochs"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model": model.state_dict()}, epoch_dir / f"{epoch:04d}.pt")

        if (
            not args.no_samples
            and sample_prompts
            and should_run_every(epoch, args.sample_every)
        ):
            from viz_data import save_epoch_samples

            save_epoch_samples(
                model,
                run_dir,
                epoch,
                device,
                amp=amp,
                inner_iters=val_inner,
                prompts=sample_prompts,
                tokenizer=tokenizer,
                max_new_tokens=args.viz_max_new_tokens,
                batch_size=args.viz_batch_size,
            )

        print(
            f"epoch {epoch}: train_loss={train_stats.loss:.4f} val_loss={val_stats.loss:.4f} "
            f"val_ppl={val_stats.ppl:.2f} val_bpc={val_stats.bpc}",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train looped causal LM on TinyStories")
    p.add_argument("--preset", choices=sorted(PRESETS))
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--inner-iters", type=int, default=None)
    p.add_argument("--loss-iters", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--num-kv-heads", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--no-weight-tying", action="store_true")
    p.add_argument("--naive-attn", action="store_true")
    p.add_argument("--val-inner-iters", type=int, default=None)
    p.add_argument("--val-loss-iters", type=int, default=None, help="Val loss loop window (default: loss-iters)")
    p.add_argument("--train-batch-size", type=int, default=4)
    p.add_argument("--val-batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batches-per-epoch", type=int, default=200)
    p.add_argument("--val-batches", type=int, default=50)
    p.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument(
        "--warmup-epochs",
        type=float,
        default=10.0,
        help="Linear LR warmup length in epochs (optimizer steps)",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Override warmup length in optimizer steps (default: warmup-epochs × steps/epoch)",
    )
    p.add_argument("--lr-schedule", choices=LR_SCHEDULES, default="cosine")
    p.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.0,
        help="decay floor as a fraction of base lr (0 = zero, 0.1 = 10%% of peak)",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--val-max-samples", type=int, default=10000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--adam-lr", type=float, default=3e-4)
    p.add_argument("--no-muon", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-val-bpc", action="store_true", help="Skip BPC during validation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-epochs", action="store_true")
    p.add_argument("--save-every", type=int, default=1, help="Write last.pt every N epochs")
    p.add_argument("--no-samples", action="store_true", help="Skip epoch-end sample generation")
    p.add_argument("--sample-every", type=int, default=1, help="Generate samples every N epochs")
    p.add_argument("--viz-samples", type=int, default=6, help="Prompts to sample each epoch")
    p.add_argument("--viz-max-new-tokens", type=int, default=64)
    p.add_argument("--viz-batch-size", type=int, default=4, help="Sample prompts per chunk")
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def sync_args_from_model(args: argparse.Namespace, model: LoopedCausalLM) -> None:
    cfg = model.cfg
    args.dim = cfg.dim
    args.num_blocks = cfg.num_blocks
    args.inner_iters = cfg.inner_iters
    args.loss_iters = cfg.loss_iters
    args.num_heads = cfg.num_heads
    args.num_kv_heads = cfg.num_kv_heads
    args.max_seq_len = cfg.max_seq_len
    args.vocab_size = cfg.vocab_size
    if getattr(args, "val_loss_iters", None) is None:
        args.val_loss_iters = cfg.loss_iters


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    require_token_bins()
    meta = load_meta()
    args.tokenizer_hash = meta["tokenizer_hash"]
    args.model = "looped-decoder"
    model = build_model_from_args(args)
    sync_args_from_model(args, model)
    del model

    run_dir = args.runs_dir / (args.run_id or make_run_id())
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(run_dir, args)
    (run_dir / "manifest.json").write_text(json.dumps({"sample_epochs": []}, indent=2))
    print(f"run_dir={run_dir}")
    train_run(run_dir, args)


if __name__ == "__main__":
    main()
