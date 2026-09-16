# Jetson Orin Nano

## Build

```bash
cd jetson
docker compose build
```

## Recommended workflow

1. Desktop: `download-dataset` → `train-tokenizer` → `tokenize-cache`
2. Rsync `data/tokenizer/` + `data/tinystories/{train,val}.bin` + `meta.json` to Jetson
3. Jetson:

```bash
docker compose run --rm train \
  --preset tiny --seq-len 256 --train-batch-size 2 --grad-accum-steps 16 \
  --loss-iters 2 --epochs 10
docker compose run --rm generate runs/<run-id> --prompt "Once upon a time"
docker compose up viz
```

Port-forward viz: `ssh -L 8000:localhost:8000 jetson`
