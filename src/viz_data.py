from __future__ import annotations

import json
from pathlib import Path

import torch

from amp import AmpConfig, autocast_context
from generate import DEFAULT_PROMPTS, generate_greedy
from model import LoopedCausalLM
from tokenizer import TextTokenizer


def load_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {"sample_epochs": []}


def save_manifest(run_dir: Path, manifest: dict) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def save_epoch_samples(
    model: LoopedCausalLM,
    run_dir: Path,
    epoch: int,
    device: torch.device,
    *,
    amp: AmpConfig | None = None,
    inner_iters: int | None = None,
    prompts: list[str] | None = None,
) -> None:
    amp = amp or AmpConfig(enabled=False, dtype=None, scaler=None)
    tokenizer = TextTokenizer.load()
    inner = inner_iters or model.cfg.inner_iters
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for prompt in prompts or DEFAULT_PROMPTS:
        prompt_ids = tokenizer.encode(prompt)
        ids = generate_greedy(
            model,
            prompt_ids,
            max_new_tokens=64,
            inner_iters=inner,
            eos_id=tokenizer.eos_id,
            device=device,
            amp=amp,
        )
        completion = tokenizer.decode(ids[len(prompt_ids) :])
        rows.append({"prompt": prompt, "completion": completion})
    out = samples_dir / f"epoch_{epoch:04d}.json"
    out.write_text(json.dumps(rows, indent=2))
    manifest = load_manifest(run_dir)
    epochs = set(manifest.get("sample_epochs", []))
    epochs.add(epoch)
    manifest["sample_epochs"] = sorted(epochs)
    save_manifest(run_dir, manifest)
