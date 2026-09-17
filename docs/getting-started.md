# Getting started

All commands run from `jetson/` on the device. See [jetson/README.md](../jetson/README.md) for build options and details.

```bash
cd jetson
docker compose build
docker compose run --rm download
docker compose run --rm train-tokenizer
docker compose run --rm tokenize
docker compose run --rm train \
  --preset tiny --seq-len 256 --train-batch-size 2 --grad-accum-steps 16 \
  --loss-iters 2 --epochs 10
docker compose run --rm generate runs/<run-id> --prompt "Once upon a time"
docker compose up viz
```
