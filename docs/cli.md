# CLI reference

| Command | Purpose |
|---------|---------|
| `download-dataset` | Fetch TinyStories into `data/tinystories/raw/` |
| `train-tokenizer` | Train BPE → `data/tokenizer/` (default vocab 8192) |
| `tokenize-cache` | Encode corpus → memmap `.bin` shards |
| `train` | Main training loop |
| `resume` | Continue from `runs/<id>/last.pt` |
| `generate` | Text generation |

## `train-tokenizer` flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--vocab-size` | `8192` | BPE vocabulary size |
| `--max-docs` | all | Cap train stories used for BPE training |
| `--raw-dir` | `data/tinystories/raw/` | Downloaded corpus location |
| `--output-dir` | `data/tokenizer/` | Output dir for `tokenizer.json` + `meta.json` |

Run `docker compose run --rm train-tokenizer --help` from `jetson/` for the full list.

## Key `train` flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--preset` | — | `tiny` / `small` / `medium` |
| `--num-blocks` | preset | Decoder blocks in one loop pass |
| `--inner-iters` | preset | Loop count |
| `--loss-iters` | all iters | Supervise last T loop iterations |
| `--no-input-injection` | off | Ablate Y ← M(Y) |
| `--no-weight-tying` | off | Separate embed + lm_head |
| `--val-inner-iters` | same as train | Override loop count for val / generate |
| `--muon-lr` | 0.02 | Muon group LR |
| `--adam-lr` | 3e-4 | AdamW group LR |
| `--no-muon` | off | AdamW-only |
| `--seed` | 0 | Reproducibility |

Run `docker compose run --rm train --help` from `jetson/` for the full list.
