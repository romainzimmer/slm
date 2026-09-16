# slm

Looped decoder-only transformer trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). Hand-written architecture (RoPE, RMSNorm, SwiGLU, GQA) with Muon + AdamW optimization.

## Quick start

```bash
uv sync
uv run download-dataset
uv run train-tokenizer
uv run tokenize-cache
uv run train --preset tiny --max-samples 50000 --epochs 10
uv run generate runs/<run-id> --prompt "Once upon a time"
```

See [docs/getting-started.md](docs/getting-started.md), [docs/cli.md](docs/cli.md), [docs/runs.md](docs/runs.md), and [jetson/README.md](jetson/README.md).

## Viz

```bash
uv run python -m http.server 8000
# open http://localhost:8000/viz/
```
