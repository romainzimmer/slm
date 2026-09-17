from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from amp import AmpConfig, autocast_context, resolve_amp
from model import KVCache, LoopedCausalLM
from tokenizer import TextTokenizer
from train import build_model_from_args, require_run_args, resolve_checkpoint_target


DEFAULT_TEMPERATURE = 0.8

DEFAULT_PROMPTS = [
    "Once upon a time",
    "The little girl",
    "The little boy wanted to",
    "One day, a small dog",
    "Tom and Lily were playing",
    "In the big green forest,",
    "Mom said,",
    "The cat looked at the",
    "It was a rainy day and",
    "The happy bunny found a",
    "At school, the children",
    "The brave knight saw a",
]


def pick_next_token(logits: torch.Tensor, temperature: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.inference_mode()
def forward_at_pos(
    model: LoopedCausalLM,
    token_id: int,
    *,
    pos: int,
    inner_iters: int,
    device: torch.device,
    amp: AmpConfig,
    cache: KVCache | None,
) -> torch.Tensor:
    idx = torch.tensor([[token_id]], dtype=torch.long, device=device)
    with autocast_context(device, amp):
        out = model(idx, inner_iters=inner_iters, kv_cache=cache, pos=pos)
    return out.logits[:, -1, :]


@torch.inference_mode()
def generate_greedy(
    model: LoopedCausalLM,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    inner_iters: int,
    eos_id: int,
    device: torch.device,
    amp: AmpConfig,
    temperature: float = 0.0,
) -> list[int]:
    """Autoregressive decode with per-loop KV cache."""
    model.eval()
    cache = KVCache(model.cfg.max_seq_len)
    ids = list(prompt_ids)
    if not ids:
        return ids
    logits = None
    for pos, token_id in enumerate(ids):
        logits = forward_at_pos(
            model, token_id, pos=pos, inner_iters=inner_iters, device=device, amp=amp, cache=cache
        )
    for _ in range(max_new_tokens):
        assert logits is not None
        next_id = pick_next_token(logits, temperature)
        ids.append(next_id)
        if next_id == eos_id:
            break
        logits = forward_at_pos(
            model,
            next_id,
            pos=len(ids) - 1,
            inner_iters=inner_iters,
            device=device,
            amp=amp,
            cache=cache,
        )
    return ids


@torch.inference_mode()
def generate_full_forward(
    model: LoopedCausalLM,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    inner_iters: int,
    eos_id: int,
    device: torch.device,
    amp: AmpConfig,
    temperature: float = 0.0,
) -> list[int]:
    model.eval()
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        with autocast_context(device, amp):
            out = model(idx, inner_iters=inner_iters)
            logits = out.logits[:, -1, :]
        next_id = pick_next_token(logits, temperature)
        ids.append(next_id)
        if next_id == eos_id:
            break
    return ids


def load_checkpoint_model(target: Path, device: torch.device) -> tuple[LoopedCausalLM, dict]:
    _, checkpoint_path = resolve_checkpoint_target(target)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    run_args = require_run_args(ckpt, source_name=str(checkpoint_path))
    model = build_model_from_args(argparse.Namespace(**run_args)).to(device)
    model.load_state_dict(ckpt["model"])
    return model, run_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint")
    parser.add_argument("target", type=Path)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPTS[0])
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--inner-iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, run_args = load_checkpoint_model(args.target, device)
    tokenizer = TextTokenizer.load()
    inner = args.inner_iters or int(
        run_args.get("val_inner_iters") or run_args.get("eval_inner_iters") or run_args["inner_iters"]
    )
    amp = resolve_amp(device, enabled=False)
    prompt_ids = tokenizer.encode(args.prompt)
    gen_fn = generate_full_forward if args.no_cache else generate_greedy
    ids = gen_fn(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        inner_iters=inner,
        eos_id=tokenizer.eos_id,
        device=device,
        amp=amp,
        temperature=args.temperature,
    )
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
