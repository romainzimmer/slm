# CLI reference

| Command | Purpose |
|---------|---------|
| `download-dataset` | Fetch TinyStories into `data/tinystories/raw/` |
| `train-tokenizer` | Train BPE → `data/tokenizer/` (default vocab 8192) |
| `tokenize-cache` | Encode corpus → memmap `.bin` shards |
| `train` | Main training loop |
| `resume` | Continue from `runs/<id>/last.pt` |
| `generate` | Text generation |

## `download-dataset` flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--raw-dir` | `data/tinystories/raw/` | Output directory for downloaded splits |

## `train-tokenizer` flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--vocab-size` | `8192` | BPE vocabulary size |
| `--max-docs` | all | Cap train stories used for BPE training |
| `--raw-dir` | `data/tinystories/raw/` | Downloaded corpus location |
| `--output-dir` | `data/tokenizer/` | Output dir for `tokenizer.json` + `meta.json` |

## `tokenize-cache` flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--raw-dir` | `data/tinystories/raw/` | Downloaded corpus location |
| `--tokenizer-dir` | `data/tokenizer/` | Trained tokenizer directory |
| `--output-dir` | `data/tinystories/` | Output dir for `train.bin`, `val.bin`, `meta.json` |

## `train` flags

### Model

| Flag | Default | Meaning |
|------|---------|---------|
| `--preset` | `small` | `tiny` / `small` / `medium` |
| `--dim` | preset | Hidden dimension |
| `--num-blocks` | preset | Decoder blocks in one loop pass |
| `--inner-iters` | preset | Loop count |
| `--loss-iters` | all iters | Supervise last T loop iterations |
| `--num-heads` | preset | Query attention heads |
| `--num-kv-heads` | preset | Key/value heads (GQA) |
| `--max-seq-len` | `1024` | Max context window (RoPE, mask, KV cache) |
| `--seq-len` | `512` | Training chunk length |
| `--vocab-size` | `8192` | Embedding / output vocabulary size |
| `--no-weight-tying` | off | Separate embed + lm_head |
| `--naive-attn` | off | Explicit softmax attention (tests) |
| `--val-inner-iters` | same as train | Override loop count for val / generate |
| `--val-loss-iters` | same as `--loss-iters` | Val loss loop window |

### Data & batches

| Flag | Default | Meaning |
|------|---------|---------|
| `--train-batch-size` | `4` | Training batch size |
| `--val-batch-size` | `4` | Validation batch size |
| `--grad-accum-steps` | `8` | Gradient accumulation steps |
| `--grad-clip` | `1.0` | Max gradient norm |
| `--batches-per-epoch` | `200` | Optimizer steps per epoch |
| `--val-batches` | `50` | Validation batches per epoch |
| `--val-every` | `1` | Run validation every N epochs |
| `--max-samples` | all | Cap training dataset samples |
| `--val-max-samples` | `10000` | Cap validation dataset samples (fixed windows) |
| `--num-workers` | `0` | DataLoader worker processes |
| `--no-val-bpc` | off | Skip BPC during validation |

### Schedule & optimizer

| Flag | Default | Meaning |
|------|---------|---------|
| `--epochs` | `3` | Total training epochs |
| `--max-steps` | `epochs × ⌈batches-per-epoch / grad-accum⌉` | LR decay horizon (optimizer steps) |
| `--warmup-epochs` | `10` | Linear LR warmup length in epochs (optimizer steps) |
| `--warmup-steps` | `warmup-epochs × ⌈batches-per-epoch / grad-accum⌉` | Optional override in optimizer steps |
| `--lr-schedule` | `cosine` | Post-warmup schedule: `cosine`, `linear`, or `constant` |
| `--min-lr-ratio` | `0.0` | Decay floor as fraction of peak LR (`0.1` = 10% of `--muon-lr` / `--adam-lr`) |
| `--muon-lr` | `0.02` | Peak Muon group LR |
| `--adam-lr` | `3e-4` | Peak AdamW group LR |
| `--no-muon` | off | AdamW-only |
| `--no-amp` | off | Disable mixed precision |

### Run output

| Flag | Default | Meaning |
|------|---------|---------|
| `--seed` | `0` | Reproducibility |
| `--save-epochs` | off | Save per-epoch weight snapshots |
| `--save-every` | `1` | Write `last.pt` every N epochs (`best.pt` always on improvement) |
| `--no-samples` | off | Skip epoch-end sample generation |
| `--sample-every` | `1` | Generate samples every N epochs |
| `--viz-samples` | `6` | Prompts to sample (fixed subset from defaults) |
| `--viz-max-new-tokens` | `64` | Tokens per sample prompt |
| `--viz-batch-size` | `4` | Sample prompts per chunk |
| `--viz-temperature` | `0.8` | Sampling temperature for epoch-end samples (`0` = greedy) |
| `--runs-dir` | `runs/` | Parent directory for run folders |
| `--run-id` | auto | Run folder name (timestamp + random suffix) |
| `--device` | `cuda` if available else `cpu` | Training device |

## `resume` flags

| Flag | Default | Meaning |
|------|---------|---------|
| `run_dir` | — | Path to existing run (positional) |
| `--epochs` | — | New total epoch count (required; must exceed completed epoch) |
| `--device` | `cuda` if available else `cpu` | Training device |

Resume reuses all hyperparameters stored in the checkpoint (`args` in `last.pt`), including LR schedule settings. New train flags missing from older checkpoints use current defaults.

## `generate` flags

| Flag | Default | Meaning |
|------|---------|---------|
| `target` | — | Run directory or `.pt` checkpoint (positional) |
| `--prompt` | `Once upon a time` | Text prompt |
| `--max-new-tokens` | `100` | Tokens to generate after the prompt |
| `--inner-iters` | checkpoint value | Loop count override |
| `--seed` | `0` | Reproducibility |
| `--temperature` | `0.8` | Sampling temperature (`0` = greedy argmax) |
| `--no-cache` | off | Disable KV cache (full-prefix forward each step) |
| `--device` | `cuda` if available else `cpu` | Inference device |
