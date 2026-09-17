# slm

Looped decoder-only transformer trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). Hand-written architecture (RoPE, RMSNorm, SwiGLU, GQA) with Muon + AdamW optimization.

## Quick start

```bash
cd jetson
docker compose build
docker compose run --rm download
docker compose run --rm train-tokenizer
docker compose run --rm tokenize
docker compose run --rm train --preset tiny --epochs 10
docker compose run --rm generate runs/<run-id> --prompt "Once upon a time"
```

See [jetson/README.md](jetson/README.md), [docs/getting-started.md](docs/getting-started.md), [docs/cli.md](docs/cli.md), and [docs/runs.md](docs/runs.md).

## Viz

```bash
cd jetson
docker compose up viz
# open http://localhost:8000/viz/
```
