# Getting started

## Install

```bash
uv sync
uv run pytest
```

## Data pipeline

```bash
uv run download-dataset
uv run train-tokenizer
uv run tokenize-cache
```

Training reads only `data/tinystories/{train,val}.bin` — no on-the-fly tokenization.

## Train

```bash
uv run train --preset tiny --seq-len 512 --train-batch-size 4 --grad-accum-steps 8 --epochs 3
```

## Generate

```bash
uv run generate runs/<run-id> --prompt "Once upon a time"
```

## Jetson

Pre-tokenize on desktop, rsync `data/tokenizer/` + `data/tinystories/*.bin` to the device, then see [jetson/README.md](../jetson/README.md).
