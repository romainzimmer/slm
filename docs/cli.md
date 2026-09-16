# CLI reference

| Command | Purpose |
|---------|---------|
| `download-dataset` | Fetch TinyStories into `data/tinystories/raw/` |
| `train-tokenizer` | Train BPE → `data/tokenizer/` (default vocab 8192) |
| `tokenize-cache` | Encode corpus → memmap `.bin` shards |
| `train` | Main training loop |
| `resume` | Continue from `runs/<id>/last.pt` |
| `generate` | Text generation |

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

Run `uv run train --help` for the full list.
