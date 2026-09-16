# Setup plan: TinyStories SLM on Jetson Orin Nano

Experiment repo for training and running small **decoder-only transformers** on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) with PyTorch. Target hardware: **Jetson Orin Nano** (8 GB unified memory). Repo layout and tooling follow `[../discrete-reasoning](../discrete-reasoning)`.

---

## Goals

- Train from scratch on TinyStories with a **looped decoder-only transformer** ([Yang et al., 2024](https://arxiv.org/abs/2311.12424)): shared blocks, **input injection** $Y \leftarrow M(Y+P)$, unrolled for `inner_iters`, plus **RoPE**, **RMSNorm**, **SwiGLU FFN**, causal self-attention.
- **Architecture is hand-written in `src/model.py`** so blocks, norms, RoPE, and attention wiring are easy to tweak — PyTorch provides ops/kernels, not a bundled `Transformer` module.
- Keep the codebase small, testable, and CLI-driven — same ergonomics as discrete-reasoning (`uv sync`, `uv run train`, Docker on Jetson).
- Fit training and inference on Orin Nano via modest model sizes, AMP, gradient accumulation, and sensible defaults.
- Reproducible runs under `runs/<run-id>/` with checkpoints, metrics history, and optional generation samples.

Non-goals for v1: distributed training, FlashAttention dependency, HuggingFace `transformers` model classes, fine-tuning other datasets, web UI beyond a static viz page.

---

## Implementation strategy (hybrid)

Same philosophy as `[../discrete-reasoning/src/model.py](../discrete-reasoning/src/model.py)`: **own the architecture, delegate the matmuls**.

### Write explicitly (experiment surface)

All of this lives in `src/model.py` (and small helpers if needed). These are the files you edit when tweaking architecture:

- `ModelConfig`, `LoopedCausalLM`, `DecoderBlock`
- RMSNorm, RoPE (cos/sin buffers + Q/K rotation)
- Q/K/V/O projections, GQA head layout, causal mask
- SwiGLU FFN wiring (gate / up / down)
- Residual stream, final norm, LM head; **weight tying on by default** (embed ↔ lm_head)
- KV cache for inference

Keep blocks readable: pre-norm → attn → residual → pre-norm → FFN → residual. No hidden layers inside framework abstractions.

### Reuse from PyTorch (primitives & training)


| Use PyTorch for      | Examples                                                        |
| -------------------- | --------------------------------------------------------------- |
| Parameterized layers | `nn.Linear`, `nn.Embedding`, `nn.Parameter`                     |
| Attention matmuls    | `F.scaled_dot_product_attention` (default on CUDA)              |
| Activations / loss   | `F.silu`, `F.cross_entropy`                                     |
| Training infra       | `Muon` + aux `AdamW`, `GradScaler`, `autocast`, `DataLoader`, checkpoint I/O |


Attention flow: **your code** projects Q/K/V and applies RoPE → **PyTorch** runs the masked softmax × V (via SDPA).

### Avoid


| Don't use                                             | Why                                       |
| ----------------------------------------------------- | ----------------------------------------- |
| `nn.TransformerEncoderLayer`, `nn.TransformerDecoder` | Opaque layout, hard to modify             |
| HuggingFace `LlamaModel`, `GPT2Model`, etc.           | Config/inheritance fights custom blocks   |
| Custom CUDA kernels / `flash-attn` in v1              | Distraction from architecture experiments |
| Reimplementing GEMM, softmax, LayerNorm kernels       | No research value                         |


### Attention backends

Support two paths behind a flag (e.g. `--naive-attn` or `ModelConfig.use_naive_attn`):

1. **Naive** — explicit `(Q @ K.T) * scale → mask → softmax → @ V`. Use in tests to verify RoPE, masking, and GQA reshaping.
2. **SDPA** (default for train/generate on CUDA) — `F.scaled_dot_product_attention` with `attn_mask`; gets fused/mem-efficient backends on Jetson without hiding block structure.

Both must share the same Q/K/V projection and RoPE code path so tweaks apply to both.

### Code sketch

```python
def apply_M(h, blocks, cos, sin, mask, *, naive_attn):
    for block in blocks:
        h = block(h, cos, sin, mask, naive_attn=naive_attn)
    return h

class LoopedCausalLM(nn.Module):
    def forward(self, idx, *, inner_iters: int, naive_attn: bool):
        x = self.embed(idx)          # P
        h = torch.zeros_like(x)      # Y_0 = 0  (paper Eq. before Sec 4.1)
        cos, sin, mask = self.rope_cache(...)
        losses = []
        for t in range(1, inner_iters + 1):
            h = apply_M(h + x, self.blocks, cos, sin, mask, naive_attn=naive_attn)  # Y_t = M(Y_{t-1} + P)
            if self.training and self.should_supervise(t, inner_iters):
                losses.append(self.head(self.norm(h)))
        logits = self.head(self.norm(h))
        return logits, losses
```

Paper Sec 4.1: $Y_{t+1} = M(Y_t + P)$ with $Y_0 = 0$ — **not** $Y \leftarrow M(Y) + P$ and **not** embed-once-then-loop without re-adding $P$.

---

## Reference repo structure (discrete-reasoning)

Mirror these conventions:

```
slm/
  pyproject.toml          # uv, hatchling, entry points
  README.md
  LICENSE
  src/                      # flat modules, imported as top-level (pythonpath = src)
    model.py
    train.py
    eval.py
    resume.py
    generate.py
    download.py
    train_tokenizer.py
    dataset.py
    tokenizer.py
    amp.py
    optimizer.py          # Muon vs AdamW param groups
    muon.py               # vendored from KellerJordan/Muon (MIT)
    viz_data.py           # manifest + epoch samples for viz
    tokenize_cache.py
  tests/
  docs/
    getting-started.md
    cli.md
    runs.md
    model.md
  jetson/
    Dockerfile
    docker-compose.yml
    README.md
  data/                     # gitignored, downloaded artifacts
  runs/                     # gitignored, training outputs
  viz/                      # optional static charts / sample viewer
  .github/workflows/ci.yml
```

Patterns to reuse verbatim where possible:

- **Packaging**: hatchling wheel with `src/` mapped to package root; CLI via `[project.scripts]`.
- **Runs**: timestamp + short hash run id; `history.json`, `manifest.json`, `last.pt`, `best.pt`, optional `epochs/`.
- **Jetson**: bind-mount repo; compose services: `download`, `train-tokenizer`, `tokenize`, `train`, `resume`, `eval`, `generate`, `viz`.
- **Reproducibility**: `--seed` (default `0`) for init, data sampling, generate; store in `history.json` `args`.
- **Testing**: pytest with `pythonpath = ["src"]`; smoke tests for model forward, dataset batching, checkpoint round-trip.
- **Platform pins**: same `numpy<2` / `torch<2.3` guard for Intel Mac in `pyproject.toml`.

---

## Model architecture

**Looped recurrence** from [Giannou et al. (2023)](https://arxiv.org/abs/2311.12424) / [Yang et al. ICLR 2024](https://arxiv.org/abs/2311.12424) — applied here to **causal LM on TinyStories**, not the paper’s in-context learning (ICL) task. Block internals are modern LLaMA-style (RoPE, RMSNorm, SwiGLU), not the paper’s GPT-2/Garg setup.

### Paper vs this repo

| | Paper (Sec 3–4) | This repo |
|--|-----------------|-----------|
| **Task** | In-context learning on synthetic prompts $(x_i, f(x_i), …)$ | Next-token LM on TinyStories |
| **Loss** | MSE on predicted $f(x_{i+1})$ | Token CE (shifted labels) |
| **Backbone** | GPT-2 decoder (Garg et al.), $L$ layers in one $M$ | RoPE + RMSNorm + SwiGLU blocks |
| **Loop** | $Y_t = M_\theta(\… M_\theta(Y_0+P)+P \…)+P$, $Y_0=0$ | Same recurrence on embedded tokens $P$ |
| **Input injection** | **Required** — add $P$ after each $M$ (Sec 4.1) | **Default on**; `--no-input-injection` ablation |
| **$L$ vs `num_blocks`** | $M$ = one $L$-layer stack per loop | `num_blocks` = $L$; one loop = full block stack |
| **$b$ vs `inner_iters`** | Train/infer loop count $b$ (e.g. 20) | `--inner-iters` (= $b$) |
| **$T$ vs supervision** | Loss on iterations $t \in [\max(b-T,0), b]$ only (Eq. 1, TBPTT-style) | `--loss-iters` (= $T$); default = all iters if unset |
| **$b$ schedule** | Increase $b$ during training (Sec 5.1) | Optional `--loop-schedule` phase 6 |
| **Optimizer** | Adam, lr 1e-4, no WD | Muon + AdamW (project choice) |
| **Extrapolation** | Stable fixed point beyond trained $b$ at inference | `--eval-inner-iters` sweep to probe |

Reference implementation: [Leiay/looped_transformer](https://github.com/Leiay/looped_transformer).

### Looped transformer (default)

**Paper recurrence** (input injection — Sec 4.1): let $P$ = token embeddings, $Y_0 = 0$, $M$ = one pass through all `num_blocks`:

```
P = embed(tokens)
Y = 0
repeat inner_iters times:        # b in the paper
    Y = M(Y + P)                 # Sec 4.1 — add P before each M, not after
logits = lm_head(norm(Y))
```

| Concept | Plan flag | Paper symbol |
|---------|-----------|--------------|
| Layers in one $M$ | `--num-blocks` | $L$ |
| Loop count | `--inner-iters` | $b$ |
| Supervision window | `--loss-iters` | $T$ (last $T$ iterations of $b$) |
| Effective depth | `num_blocks × inner_iters` | $\approx$ depth; paper notes looped needs **more** iters than stacked layers for parity |

**Without input injection** ($Y \leftarrow M(Y)$ only): performance collapses beyond trained iterations (paper Fig 3). Keep `--input-injection` default **on**.

**From discrete-reasoning (not in paper):**

| Borrow | Skip for v1 LM |
|--------|----------------|
| `--inner-iters`, `--num-blocks` CLI naming | Outer commit loop |
| Multi-iter loss / deep supervision (align with paper $T$, not DR halt) | Halt head |
| `eval --sweep` on `inner_iters` | Detached cell memory |
| `history.json` logs loop config | GT reveal / curriculum |

**Training loss (LM + paper-style multi-iter supervision):**

- Compute CE at loop iterations $t \in [b-T, b]$ (if `--loss-iters T`; default $T=b$ = all iterations). Same shifted targets every $t$.
- Mean CE over supervised iterations (paper Eq. 1 spirit; truncated window saves memory — use **`--loss-iters`** smaller on Orin before disabling all supervision).
- **`--no-deep-supervision`**: CE on final iteration only ($T=1$).
- No halt loss.

**Test-time compute:** `eval` / `generate` accept `--inner-iters` (paper: can exceed trained $b$ toward fixed point). Sweep for ablations.

### DecoderBlock (one layer in the shared stack)

Repeat `num_blocks` modules in `nn.ModuleList` — **not** `num_blocks × inner_iters` unique modules.


| Component | Choice                                                                                            |
| --------- | ------------------------------------------------------------------------------------------------- |
| Norm      | **RMSNorm** (pre-norm), `eps=1e-6`                                                                |
| Attention | Multi-head **causal self-attention**                                                              |
| Positions | **RoPE** on Q/K (no absolute position embeddings)                                                 |
| FFN       | **SwiGLU** (`silu(gate) * up` → down), hidden dim = `round(8/3 * dim)` rounded to multiple of 256 |
| Residual  | Pre-norm + residual around attn and FFN                                                           |
| Biases    | Off on Linear layers (LLaMA-style)                                                                |
| Dropout   | 0 in v1 (TinyStories trains fine without it at small scale)                                       |


### Model-level

- **`TokenEmbedding` + `lm_head` weight tying: default on** — single `(vocab_size, dim)` matrix; `--no-weight-tying` for ablations. Orthogonal to looped depth (looping shares block weights; tying shares embed/head weights).
- Final **RMSNorm** before logits (after last inner iteration).
- Causal mask built once (or cached) up to `max_seq_len`.
- **`max_seq_len` ≥ `--seq-len`**: presets set both (e.g. `max_seq_len=1024`, train/eval `seq_len=512`). RoPE buffers sized to `max_seq_len`.
- `ModelConfig` dataclass: `vocab_size=8192`, `dim`, `num_blocks`, `inner_iters`, `num_heads`, `num_kv_heads`, `max_seq_len`, `seq_len`, `rope_theta` (10000), `input_injection` (true), `loss_iters` (default `inner_iters`), `deep_supervision` (true), `tie_weights` (true).

### Size presets (Orin-friendly starting points)

Params scale with `num_blocks`; `inner_iters` adds compute/memory, not weights.


| Name     | dim | num_blocks | inner_iters | eff. depth | heads | kv_heads | ~params | Notes                          |
| -------- | --- | ---------- | ----------- | ---------- | ----- | -------- | ------- | ------------------------------ |
| `tiny`   | 256 | 2          | 4           | 8          | 4     | 4        | ~8M     | pytest default                 |
| `small`  | 512 | 2          | 6           | 12         | 8     | 4        | ~28M    | TinyStories ballpark           |
| `medium` | 768 | 3          | 6           | 18         | 12    | 4        | ~75M    | Desktop training; Orin inference-only at low `seq_len` |


**GQA** on `small` / `medium` (`kv_heads=4`); `tiny` uses MHA (`heads == kv_heads`).

**Jetson OOM:** lower `--inner-iters` or `--train-batch-size` before dropping `num_blocks`. Prefer **`--no-deep-supervision`** on Orin before other cuts. Activation checkpointing inside the inner loop: Phase 5 if still OOM.

### RoPE implementation notes

- Precompute `cos/sin` buffers for `[max_seq_len, head_dim]`.
- Apply to Q and K after projection, before attention.
- Support `rope_theta` and optional **YaRN** later; not required for v1 (TinyStories uses short context).

### KV cache + looping (v1 spec)

Each decode step runs **`inner_iters` × `num_blocks`** forward passes on the growing prefix. v1 scheme:

1. **Cache key:** `(loop_iter, block_idx)` — separate K/V tensors per block **per inner iteration** (same block weights, different activations each loop pass).
2. **One new token per step:** append position `t`; each `(loop_iter, block)` reads/writes cache slot `t` only for that layer pass.
3. **`generate.py` must match `test_generate.py`:** greedy decode with cache == full forward (no cache) on fixed prefix, for several `inner_iters` values.

Document implementation in `docs/model.md`. Do not ship `generate.py` until this test passes.

**Perf note:** decode cost scales with `inner_iters` × `num_blocks`; `--sweep` on `inner_iters` trades quality vs tokens/sec.

---

## Tokenizer & dataset

### Tokenizer (`src/tokenizer.py`, `src/train_tokenizer.py`)

**Default vocab size: 8192** (fixed for v1; `--vocab-size` override only for ablations). Keeps embed + lm_head small (~8k × dim instead of ~50k with GPT-2), which matters for `tiny` presets and Orin Nano.

**Train** (`train-tokenizer` CLI, wraps HuggingFace `tokenizers` BPE — training lib only, not `transformers` models):

- Input: TinyStories train split from `data/tinystories/raw/` (via `download-dataset`).
- Output: `data/tokenizer/tokenizer.json` + `data/tokenizer/meta.json` (vocab size, special token ids, byte/token stats, training sample count).
- Special tokens: `<|endoftext|>` as **EOS** (story separator + generation stop).
- **v1 training:** random memmap slices only — **no padding**, no PAD token in loss. Do not reuse EOS id as PAD (add `<|pad|>` later if variable-length batches).
- Train on a large subset (default: all train stories, or `--max-docs` cap for fast iteration).

**`tokenizer_hash`:** SHA256 hex of `tokenizer.json` bytes + `vocab_size`; stored in `data/tokenizer/meta.json` and copied into `data/tinystories/meta.json`. `train` / `tokenize-cache` refuse mismatch.

**Runtime wrapper** (`src/tokenizer.py`):

- Load `tokenizer.json`; expose `encode(text) -> list[int]`, `decode(ids) -> str`, `vocab_size`, `eos_id`.
- `ModelConfig.vocab_size` must match tokenizer meta; serialized in every checkpoint and `history.json`.

Changing vocab or retraining tokenizer invalidates token caches and checkpoints — re-run `tokenize-cache` when `tokenizer_hash` changes.

**Disk budget (order of magnitude):** raw HF cache ~2–3 GB; `train.bin` + `val.bin` ~1–2 GB uint16 tokens; plan ~5 GB free under `data/`.

### Data splits

Use the **official HuggingFace TinyStories splits** (most common in repros — nanoGPT, etc.):

| Split | HF name | Output | Used for |
|-------|---------|--------|----------|
| Train | `train` | `train.bin` | Optimization |
| Val | `validation` | `val.bin` | Epoch metrics, `best.pt`, `eval`, `generate` sanity |

**No separate test split in v1.** HF does not ship a third held-out split; `validation` is already disjoint from `train`. That is enough for an experiment repo: val loss drives checkpoint selection and final reporting. A dedicated test set only matters for publishable benchmarks where val was used for hparam tuning — not needed here.

Document split names and token counts in `data/tinystories/meta.json`.

### Dataset (`src/dataset.py`)

- Source: HuggingFace `roneneldan/TinyStories` — **`train`** and **`validation`** splits only.
- **Training and eval read only pre-tokenized `.bin` shards** — no HF streaming or on-the-fly tokenization at train time (critical for Jetson throughput and stable `num_workers`).

**`.bin` format (v1):**

- Flat contiguous token ids: **`uint16` memmap** (vocab ≤ 65535; default 8192 fits).
- Files: `data/tinystories/train.bin`, `val.bin`.
- Sidecar `data/tinystories/meta.json`: `tokenizer_hash`, vocab size, token counts, dtype, EOS id.
- `TokenDataset`: random contiguous windows of `seq_len + 1` via memmap slice — no Python tokenization in the hot path. Windows may cross EOS boundaries (nanoGPT-style); all positions contribute to CE in v1.
- `DataLoader`: `num_workers` 1–2 on Orin (fallback **`0`** if fork issues); workers only slice memmap offsets.

**`download-dataset`:** cache raw text (or HF arrow shards) under `data/tinystories/raw/` so `tokenize-cache` on Jetson does not re-hit the network. Not used after `.bin` exists.

**`tokenize-cache`:** required before `train` / `eval`; encodes raw → `.bin`. `train` refuses to start if `.bin` missing or `tokenizer_hash` mismatch.

- Training sample: fixed-length chunks of `seq_len + 1` tokens (input = `[:-1]`, target = `[1:]`, CE on all positions).
- **Sequence packing** (phase 6): EOS-aligned packing for efficiency — not v1.

### Offline `.bin` for Jetson

Full `tokenize-cache` on TinyStories takes **hours on Orin**. Recommended workflow:

1. Desktop: `download-dataset` → `train-tokenizer` → `tokenize-cache`
2. Copy `data/tokenizer/` + `data/tinystories/{train,val}.bin` + `meta.json` to Jetson (rsync / external drive)
3. Jetson: `train` / `eval` only — skip re-tokenize unless tokenizer changes

Jetson compose still exposes `train-tokenizer` / `tokenize` for small `--max-docs` smoke tests.

### Preprocessing pipeline

```
download-dataset   →  data/tinystories/raw/          (one-time text cache)
train-tokenizer    →  data/tokenizer/tokenizer.json, meta.json
tokenize-cache     →  data/tinystories/train.bin, val.bin, meta.json
train / eval       →  read .bin only (memmap)
```

`tokenize-cache` is **mandatory** on every machine that trains (including Jetson). Commit `.bin` to external storage or rebuild locally — do not fall back to streaming.

---

## Training (`src/train.py`)

Follow discrete-reasoning's training shell:

- Argparse CLI with full config serialized into `history.json` (`model: looped-decoder`, `num_blocks`, `inner_iters`, `deep_supervision`, `muon_lr`, `adam_lr`, …).
- **Cosine LR** with linear warmup (`warmup_steps`, `max_steps` or derive from epochs × steps/epoch) — applied per optimizer group.
- **Gradient clipping** (global norm 1.0).
- **AMP** on CUDA (`src/amp.py` — reuse pattern from discrete-reasoning: bf16 if supported else fp16).
- **Gradient accumulation** (`--grad-accum-steps`) to simulate larger batch on memory-limited Orin.
- Checkpoint: `last.pt`, `best.pt` (best val loss), optional `epochs/NNNN.pt`.
- Resume: restore model, optimizer(s), scaler, step/epoch, RNG state.

### Optimizer — Muon + AdamW (default from v1)

Hybrid per [Muon](https://github.com/KellerJordan/Muon) / [Keller Jordan (2024)](https://kellerjordan.github.io/posts/muon/): **Muon on hidden 2D matrices**, **AdamW on everything else**. Not Muon-only — embeddings and norms stay on AdamW; with **weight tying**, embed/lm_head is one AdamW tensor.

| Parameters | Optimizer | Notes |
|------------|-----------|-------|
| `DecoderBlock` `Linear` weights (Q/K/V/O, gate/up/down) | **Muon** | `bias=False`; flatten conv-like dims if ever added |
| `TokenEmbedding` (+ tied lm head) | **AdamW** | Aux group |
| `RMSNorm.weight` | **AdamW** | 1D; never Muon |

Implementation:

- Vendor `src/muon.py` from [KellerJordan/Muon](https://github.com/KellerJordan/Muon) (`MuonWithAuxAdam` or equivalent dual-group wrapper).
- `src/optimizer.py`: `build_optimizer(model, cfg) -> Optimizer` — assigns params, no manual listing per layer in `train.py`.
- Default LRs (sweep in logspace on `tiny` before long Orin runs):

| Group | Default LR | Other |
|-------|------------|-------|
| Muon (hidden) | `0.02` | momentum `0.95`, Nesterov, `ns_steps=5`, weight decay `0.01` |
| AdamW (aux) | `3e-4` | betas `(0.9, 0.95)`, weight decay `0.01` on aux weights |

Muon applies built-in spectral scaling (`∝ sqrt(max(d_out, d_in))`); do not reuse a single `--lr` for both groups.

- **`--no-muon`**: ablation — AdamW on all params (discrete-reasoning-style); weight decay on matmul weights only, not norms.
- Checkpoints store both optimizer states when using Muon hybrid.
- Jetson: Muon Newton–Schulz runs in bf16 on CUDA; validate one training step on Orin in Phase 5 (memory + numerics).

References: [Muon is Scalable for LLM Training](https://arxiv.org/pdf/2502.16982v1) (Moonlight).

### Key training flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--preset` | — | `tiny` / `small` / `medium` (sets dim, num_blocks, inner_iters, heads) |
| `--num-blocks` | from preset | Unique `DecoderBlock` count |
| `--inner-iters` | from preset | Loop count $b$ (train + eval unless overridden) |
| `--loss-iters` | same as `inner_iters` | Supervise last $T$ loop iterations only (paper Eq. 1) |
| `--no-input-injection` | off | Ablate to $Y \leftarrow M(Y)$ (paper Fig 3 — diverges) |
| `--no-deep-supervision` | off | CE on final loop iter only ($T=1$) |
| `--no-weight-tying` | off | Separate embed + lm_head matrices |
| `--eval-inner-iters` | same as train | Override for val / `eval` / `generate` |
| `--muon-lr` | `0.02` | Hidden 2D matrices (Muon group) |
| `--adam-lr` | `3e-4` | Embed, norms, non-Muon params |
| `--no-muon` | off | AdamW-only ablation |
| `--seed` | `0` | Init, dataset sampling, generate |

Also: `--seq-len`, cosine LR + warmup, grad clip `1.0`, AMP (default on CUDA), `--grad-accum-steps`. Run `uv run train --help` for full list.

### Default hyperparameters

**Desktop smoke:**

```bash
uv run train \
  --preset tiny \
  --seq-len 512 \
  --train-batch-size 4 \
  --grad-accum-steps 8 \
  --batches-per-epoch 200 \
  --epochs 3 \
  --max-samples 10000
```

**Orin Nano (first e2e — use after rsync of `.bin`):**

```bash
uv run train \
  --preset tiny \
  --seq-len 256 \
  --train-batch-size 2 \
  --grad-accum-steps 16 \
  --no-deep-supervision \
  --batches-per-epoch 200 \
  --epochs 3 \
  --max-samples 10000
```

### Metrics per epoch (in `history.json`)

Per epoch in `epochs[]`:

- `train_loss`, `val_loss`, `train_ppl`, `val_ppl` (token-level)
- `train_bpc`, `val_bpc` (see below)
- `lr_muon`, `lr_adam` (after schedule step)
- `tokens/sec`, `step_time_ms`
- optional: `grad_norm`, `mem_allocated_mb`

Top-level `args`: `model`, `preset`, `num_blocks`, `inner_iters`, `deep_supervision`, `muon_lr`, `adam_lr`, `seq_len`, `seed`, `tokenizer_hash`, …

**BPC (bits per character):** on val, decode each batch's target tokens to string via tokenizer, accumulate `total_bits = sum(cross_entropy) * ln(2)`, divide by **UTF-8 character count** of decoded targets (not token count). Log `val_bpc`; same formula for train when computed.

`eval --sweep` writes top-level `eval: { inner_iters_sweep: [...] }` in `history.json`.

### Run directory layout

```
runs/<run-id>/
  history.json
  manifest.json          # sample epochs + paths for viz
  last.pt
  best.pt
  epochs/0003.pt
  samples/
    epoch_0003.json      # [{prompt, completion}, ...] for viz
  profile/
    trace.json           # optional
```

---

## Inference & eval

### `generate.py`

- Default checkpoint: `runs/<run-id>/best.pt` (or `last.pt` if `--last`).
- Args: `runs/<run-id>` or explicit `.pt` path.
- Flags: `--prompt`, `--max-new-tokens`, `--temperature`, `--top-p`, `--top-k`, `--seed`, `--inner-iters`.
- **KV cache** per [KV cache + looping](#kv-cache--looping-v1-spec) — gated on `test_generate.py`.
- Write output to stdout or append to `runs/<run-id>/samples/`.

### `eval.py`

- Val loss, perplexity, BPC on `val.bin`.
- `--inner-iters` override; **`--sweep`** one-dimensional ablation over inner iterations (discrete-reasoning-style; results in `history.json` `eval` block).
- Optional **benchmark mode**: tokens/sec, peak memory, latency for a fixed prompt length (useful on Jetson).

Entry points — see [Dependencies](#dependencies-pyprojecttoml) for full `[project.scripts]` list.

```
generate = "generate:main"
eval     = "eval:main"
```

---

## Jetson Orin Nano

Copy and adapt `jetson/` from discrete-reasoning. Reuse the same **base image**, **torch install step**, and **compose layout** so builds share layers with discrete-reasoning where possible (see below).

### Docker

- Base: JetPack 7.2.1 image (`whitesscott/l4t-jetpack:r39.2.1`) with fallback notes for JP6.
- Torch index: `TORCH_INDEX=https://download.pytorch.org/whl/cu132` (same defaults as discrete-reasoning; JP6 override via compose build args).
- Bind-mount repo; `PYTHONPATH=/workspace/src`.
- Compose services: `download`, `train-tokenizer`, `tokenize`, `train`, `resume`, `eval`, `generate`, `viz`.
- Image tag: `slm:jetson` (separate final image from `discrete-reasoning:jetson`, but shared intermediate layers — see next section).

### Docker layer sharing with discrete-reasoning

A separate repo **does not** use the same final image automatically — `docker compose build` in `slm/jetson/` produces `slm:jetson`, not `discrete-reasoning:jetson`. It also **does not** rebuild everything from scratch on the same Jetson if the early Dockerfile steps match.

**What Docker reuses (build cache on the same machine):**


| Layer                             | Shared if…                                                   |
| --------------------------------- | ------------------------------------------------------------ |
| `FROM l4t-jetpack`                | Same `BASE_IMAGE` arg                                        |
| `apt-get install python3-pip`     | Same `RUN` line                                              |
| `pip install torch --index-url …` | Same `TORCH_INDEX` and command (largest win — slow download) |


**What invalidates cache (slm-specific rebuild):**


| Layer                                  | Why                                                     |
| -------------------------------------- | ------------------------------------------------------- |
| `COPY pyproject.toml`                  | SLM adds `datasets`, `tokenizers` vs discrete-reasoning |
| `pip install .`                        | Pulls those extra deps                                  |
| `COPY src` + `pip install --no-deps .` | SLM source tree                                         |


So: first `slm` build on a machine that already built discrete-reasoning still re-downloads/reinstalls **only from `COPY pyproject.toml` onward** — not base OS or torch.

**Recommended approaches (pick one):**

1. **Same Dockerfile prefix (default)** — Copy discrete-reasoning’s `jetson/Dockerfile` verbatim through the torch `RUN`, then SLM-specific `COPY`/`pip` steps. Zero coupling; automatic cache reuse when both repos are built on the same host.
2. **Extend discrete-reasoning image (fastest SLM rebuilds)** — After `discrete-reasoning/jetson` is built once:
  ```dockerfile
   FROM discrete-reasoning:jetson
   COPY pyproject.toml README.md ./
   RUN pip3 install --no-cache-dir --default-timeout=86400 .
   COPY src ./src
   RUN pip3 install --no-cache-dir --no-deps .
  ```
   Skips base + torch entirely. Tradeoff: must build discrete-reasoning first; loose coupling between repos. Runtime still bind-mounts `slm` source, same as discrete-reasoning.
3. **Shared base image (future)** — Extract `FROM …` through torch into e.g. `jetson-pytorch:jp72` published or built once locally; both repos `FROM` it. Only worth it if you add more experiment repos.

At runtime both projects bind-mount the repo to `/workspace` and set `PYTHONPATH=/workspace/src`, so the container image is mainly **torch + pip deps** — source code always comes from the host mount.

### Memory budget & Orin defaults (8 GB unified)

| Setting | Desktop | Orin Nano (recommended) |
|---------|---------|-------------------------|
| Preset | `tiny` / `small` | **`tiny`** first e2e |
| `--seq-len` | 512 | **256** (384 if headroom) |
| `--train-batch-size` | 4–8 | **2** |
| `--grad-accum-steps` | 8 | **16** |
| Deep supervision | on (`--loss-iters` = all) | **`--loss-iters 2`** or `--no-deep-supervision` |
| `--inner-iters` | preset | preset; **−1 to −2** if OOM |
| AMP | on | on |
| `num_workers` | 1–2 | **1** (0 if unstable) |
| `medium` preset | train OK | **inference / eval only** |

Activation checkpointing inside inner loop: enable in Phase 5 if OOM persists with `--no-deep-supervision`.


### Profiling

- PyTorch profiler flags on `train` (same as discrete-reasoning).
- Optional `jetson/nsys-profile.sh` for Nsight Systems.

---

## CLI surface (`docs/cli.md`)


| Command            | Purpose                                                     |
| ------------------ | ----------------------------------------------------------- |
| `download-dataset` | Fetch TinyStories into `data/`                              |
| `train-tokenizer`  | Train BPE → `data/tokenizer/` (default `--vocab-size 8192`) |
| `tokenize-cache`   | **Required.** Encode raw corpus → memmap `.bin` shards      |
| `train`            | Main training loop                                          |
| `resume`           | Continue from `runs/<id>/last.pt`                           |
| `eval`             | Val perplexity + BPC / benchmark                            |
| `generate`         | Interactive or batch text generation                        |


Document all flags in `docs/cli.md`; keep `--help` in sync.

---

## Testing strategy

Layered validation — not a single gate. Run in order before trusting long Jetson runs.

```
1. Component unit tests     ← proves math / wiring (must pass before training)
2. Golden forward vectors   ← regression lock before architecture tweaks
3. Training smoke           ← integration on real tokens
4. Literature ballpark        ← optional confidence check, not proof of correctness
```

### Layer 1 — Component unit tests (primary correctness gate)

Compare custom code against **slow references written in the test file** (closed-form RoPE, explicit softmax attention) — not against `nn.TransformerEncoder` or HuggingFace Llama (different architecture; fights future tweaks).


| Test file           | What it proves                                                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------------------ |
| `test_rope.py`      | Q/K differ by position; `apply_rope` matches reference formula from the test                                 |
| `test_rmsnorm.py`   | Output matches explicit `x * rsqrt(mean(x²) + eps) * weight`                                                 |
| `test_attention.py` | Causal mask blocks future tokens; **naive attn == SDPA** on same Q/K/V/mask (`atol~1e-5` fp32)               |
| `test_model.py`     | looped forward; **`Y = M(Y+P)` vs no injection**; loss-iters window; shapes |
| `test_model.py`     | `loss.backward()` — finite grads, no NaN                                                                     |
| `test_generate.py`  | KV-cache decode matches full forward on same prefix                                                          |
| `test_tokenizer.py` | encode/decode round-trip; load trained `tokenizer.json` fixture                                              |
| `test_dataset.py`   | batch shapes, label shift; v1 all positions in loss (no pad mask) |
| `test_train.py`     | checkpoint save/load → identical logits; Muon+AdamW state round-trip |
| `test_optimizer.py` | every param in exactly one group; Muon vs AdamW assignment |
| `test_amp.py`       | autocast context on CPU/CUDA                                                                                 |


Implement **naive attention first**, then SDPA; parity test is the main guard that RoPE, masking, and GQA reshaping are wired correctly.

### Layer 2 — Golden vectors (regression before tweaks)

After Layer 1 passes, once per preset (start with `tiny`):

1. Fixed seed, fixed `input_ids`, eval mode, fp32.
2. Save to `tests/fixtures/tiny_forward_logits.pt` (logits tensor or scalar loss).
3. `test_golden.py` reloads fixture and asserts `torch.allclose` within tolerance.

Regenerate fixtures intentionally when architecture changes are expected (`uv run pytest --update-golden` script or documented one-off command). Commit fixtures to git — small files.

This is the main safety net while editing `src/model.py`.

### Layer 3 — Training smoke (integration)

Short run on a real token shard (CPU or CUDA):

- Few hundred optimizer steps: train loss decreases.
- Val perplexity finite and below random baseline (`~ vocab_size`).
- `generate` after a few epochs: broken-but-English prose (not repeating one token).

`test_train_smoke.py` can use `--max-samples 500` equivalent in-process; keep runtime under ~60s in CI (mark `@pytest.mark.slow` and skip on CI if needed).

### Layer 4 — Literature ballpark (optional, not strict)

Full train on Orin or desktop: compare val perplexity / sample quality **roughly** to TinyStories literature. Exact match is unrealistic (tokenizer, size, steps, data slice all differ). Use only to catch “something is very wrong”, not to prove implementation correctness.

**Do not** rely on:


| Approach                          | Why skip                                       |
| --------------------------------- | ---------------------------------------------- |
| vs `nn.TransformerEncoder`        | LayerNorm, no RoPE, different FFN              |
| vs HuggingFace Llama side-by-side | Config/detail drift; breaks when you customize |
| Dual full training runs           | 2× compute; inconclusive if hparams differ     |


One-time HF LLaMA parity on a frozen clone is optional when porting; not for ongoing architecture experiments.

### CI

```bash
uv sync --extra dev && uv run pytest
```

On `ubuntu-latest` (same as discrete-reasoning). Layers 1–2 always; Layer 3 optional/slow marker.

**Muon in CI:** `test_optimizer` uses AdamW-only or CPU-safe path; mark CUDA-only Muon step tests `@pytest.mark.cuda` and skip on CI. Golden / model tests use `--no-muon` or CPU.

### Test layout

```
tests/
  test_rope.py
  test_rmsnorm.py
  test_attention.py
  test_model.py
  test_golden.py
  test_tokenizer.py
  test_dataset.py
  test_train.py
  test_optimizer.py
  test_train_smoke.py      # optional slow
  test_generate.py
  test_amp.py
  fixtures/
    tiny_forward_logits.pt
```

---

## Documentation


| File                      | Content                                                         |
| ------------------------- | --------------------------------------------------------------- |
| `README.md`               | One-liner, quick start, link to docs                            |
| `docs/getting-started.md` | uv install, download, train, generate                           |
| `docs/model.md`           | looped transformer, paper ref, presets, KV cache scheme |
| `docs/cli.md`             | all commands and flags                                          |
| `docs/runs.md`            | run layout, history schema, viz                                 |
| `jetson/README.md`        | Docker build/train/generate on Orin; rsync `.bin` workflow |

---

## Viz (discrete-reasoning pattern)

Static SPA — no backend. Fork `../discrete-reasoning/viz/index.html`, strip sudoku-specific panels, keep run discovery + chart plumbing.

**Serve from repo root** (not `viz/` — page loads `../runs/`):

```bash
uv run python -m http.server 8000
# → http://localhost:8000/viz/
```

Jetson: `docker compose up viz` + SSH `-L 8000:localhost:8000` (same as discrete-reasoning).

### UI panels (v1)

| Panel | Data source |
|-------|-------------|
| Run picker + config `<details>` | `runs/<id>/history.json` → `args` |
| Metrics charts | `history.json` → `epochs[]`: loss, ppl, bpc, `lr_muon`, `lr_adam`, `mem_allocated_mb` |
| Sample viewer | `runs/<id>/samples/epoch_NNNN.json` (prompt + completion list) |
| Eval sweep | `history.json` → `eval.inner_iters_sweep` (ppl/bpc vs `inner_iters`) |

Drop: halt, curriculum, rating groups, sudoku trajectory player.

### `src/viz_data.py`

Called from `train.py` after validation each epoch:

- Fixed prompt list (e.g. `"Once upon a time"`, `"The little girl"`) — store in repo or `args`.
- `generate` greedy continuations; write `samples/epoch_NNNN.json`.
- Update `manifest.json` with available sample epochs (discrete-reasoning pattern).

Phase 4: write samples + manifest. Phase 5: `viz/index.html` fork.

---

## Dependencies (`pyproject.toml`)

Match discrete-reasoning’s **torch / numpy platform pins** exactly; add only SLM-specific packages on top.

```toml
dependencies = [
    "tqdm>=4.66.0",
    "huggingface-hub>=0.27.0",
    "datasets>=3.0.0",       # HF load for download + tokenize-cache only (SLM-only)
    "tokenizers>=0.20.0",    # train + load custom BPE (SLM-only)
    # --- same pins as ../discrete-reasoning/pyproject.toml ---
    "numpy<2; platform_machine == 'x86_64' and sys_platform == 'darwin'",
    "torch>=2.0.0,<2.3.0; platform_machine == 'x86_64' and sys_platform == 'darwin'",
    "torch>=2.0.0; platform_machine != 'x86_64' or sys_platform != 'darwin'",
]

[project.optional-dependencies]
dev = ["pytest"]
```

On Jetson, torch is **not** taken from these PyPI markers — the Dockerfile installs it from `TORCH_INDEX` before `pip install .`, same two-step pattern as discrete-reasoning (avoids pytorch.org hash mismatches on extra deps).

No `transformers` or `flash-attn` — model code is in-repo for easy edits. Avoid `wandb` in v1.

**Entry points** (`[project.scripts]`):

```toml
download-dataset = "download:main"
train-tokenizer  = "train_tokenizer:main"
tokenize-cache   = "tokenize_cache:main"
train            = "train:main"
resume           = "resume:main"
eval             = "eval:main"
generate         = "generate:main"
```

---

## Implementation phases

### Phase 0 — Scaffold (day 1)

- [ ] `pyproject.toml`, README, LICENSE, `.gitignore` (`data/`, `runs/`, `.venv/`)
- [ ] Empty `src/` modules + pytest smoke
- [ ] CI workflow
- [ ] `docs/setup-plan.md` (this file) → trim into getting-started once implemented

### Phase 1 — Model + unit tests (day 2–3)

- [ ] `ModelConfig`, RMSNorm, RoPE, SwiGLU, `DecoderBlock`
- [ ] `LoopedCausalLM` with **$Y \leftarrow M(Y+P)$**, `input_injection` flag, `--loss-iters` window
- [ ] Attention: naive path first, then SDPA backend
- [ ] Layer 1 tests: `test_rope`, `test_rmsnorm`, `test_attention` (naive vs SDPA + reference), `test_model`
- [ ] Generate golden fixture for `tiny` preset; add `test_golden.py`
- [ ] Param counter; `tiny` preset

### Phase 2 — Data pipeline (day 3–4)

- [ ] `download-dataset` (HF datasets)
- [ ] `train-tokenizer` — BPE on TinyStories, vocab **8192** → `data/tokenizer/`
- [ ] `tokenizer.py` wrapper (load, encode, decode)
- [ ] `tokenize-cache` → memmap `uint16` `.bin` + meta (`tokenizer_hash`, split token counts)
- [ ] `train` / `eval` hard-fail without valid `.bin`
- [ ] `test_tokenizer.py` with committed small `tests/fixtures/tokenizer.json` or train-on-tiny-subset in test

### Phase 3 — Training loop (day 4–6)

- [ ] Vendor `src/muon.py`; `src/optimizer.py` with Muon + AdamW param groups
- [ ] `train.py` with Muon hybrid (default), `--no-muon` ablation, AMP, cosine schedule, grad accum
- [ ] Run dir, checkpoints, `history.json` (incl. `lr_muon`, `lr_adam`, BPC on val)
- [ ] `resume.py`
- [ ] Layer 3: `test_train_smoke.py` (loss ↓ on token shard)
- [ ] Local smoke: 1000 steps on CPU / small CUDA

### Phase 4 — Eval & generation (day 6–7)

- [ ] `eval.py` (val ppl/BPC; `--inner-iters`, `--sweep`)
- [ ] `generate.py` with KV cache per v1 spec; **`test_generate.py` must pass before merge**
- [ ] `src/viz_data.py`: fixed-prompt samples + `manifest.json` each epoch

### Phase 5 — Jetson (day 7–8)

- [ ] `jetson/Dockerfile`, `docker-compose.yml`, README — rsync `.bin` workflow documented
- [ ] End-to-end: rsync `data/` → train `tiny` with Orin flags → `eval` → `generate`
- [ ] One Muon training step + `mem_allocated_mb` logged; activation checkpointing if OOM
- [ ] Fork `viz/index.html`; `docker compose up viz`

### Phase 6 — Polish

- [ ] Sequence packing (EOS-aligned windows)
- [ ] Nsight / profiler docs; optional `torch.compile` experiment on Orin
- [ ] Optional Layer 4: full train vs TinyStories literature ballpark
- [ ] Deep supervision on Orin once checkpointing lands (optional)

---

## References

- Dataset: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) on Hugging Face
- Looped architecture: [Looped Transformers are Better at Learning Learning Algorithms](https://arxiv.org/abs/2311.12424) (Yang et al., ICLR 2024); code [Leiay/looped_transformer](https://github.com/Leiay/looped_transformer)
- Optimizer: [Muon](https://github.com/KellerJordan/Muon) (Jordan et al.); [Muon is Scalable for LLM Training](https://arxiv.org/pdf/2502.16982v1)
- CLI / inner-loop patterns: [../discrete-reasoning](../discrete-reasoning) (`--inner-iters`, deep supervision, eval sweeps — MLP-Mixer sudoku, not LM)

---

## First commands (target end state)

Local:

```bash
uv sync
uv run download-dataset
uv run train-tokenizer
uv run tokenize-cache
uv run train --preset tiny --max-samples 50000 --epochs 10
uv run eval runs/<run-id>
uv run generate runs/<run-id> --prompt "Once upon a time"
uv run python -m http.server 8000   # → /viz/
```

Jetson (after rsync of `data/tokenizer/` + `data/tinystories/*.bin` from desktop):

```bash
cd jetson
docker compose build
docker compose run --rm train \
  --preset tiny --seq-len 256 --train-batch-size 2 --grad-accum-steps 16 \
  --no-deep-supervision --epochs 10
docker compose run --rm eval runs/<run-id>
docker compose run --rm generate runs/<run-id> --prompt "Once upon a time"
docker compose up viz
```
