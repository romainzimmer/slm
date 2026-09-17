# Jetson Orin Nano

Docker setup for **Jetson Orin Nano** on JetPack 7.2.1 (L4T r39.2.1). Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

All commands run from `jetson/`:

```bash
cd jetson
```

CLI flags: [docs/cli.md](../docs/cli.md). Run directories: [docs/runs.md](../docs/runs.md).

## Build

Only needed once (or when dependencies change). Source is bind-mounted from `../src`.

```bash
docker compose build
```

Override base image or PyTorch index:

```bash
BASE_IMAGE=whitesscott/l4t-jetpack:r39.2.1 \
TORCH_INDEX=https://download.pytorch.org/whl/cu132 \
docker compose build
```

Orin Nano on JetPack 6 (L4T r36.4.x):

```bash
BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
TORCH_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126 \
docker compose build
```



## Download dataset

```bash
docker compose run --rm download
```



## Train tokenizer

```bash
docker compose run --rm train-tokenizer
```



## Tokenize cache

```bash
docker compose run --rm tokenize
```

Training reads only `data/tinystories/{train,val}.bin` — no on-the-fly tokenization.

## Train

```bash
docker compose run --rm train --preset small --seq-len 256 --train-batch-size 4 --grad-accum-steps 8   --loss-iters 2 --epochs 100 --batches-per-epoch 500
```

If you hit OOM, lower `--train-batch-size`.

## Resume

```bash
docker compose run --rm resume runs/<run-id> --epochs 10
```

`--epochs` is the new total; must exceed the completed epoch.

## Generate

```bash
docker compose run --rm generate runs/<run-id> --prompt "Once upon a time"
```



## Visualize

```bash
docker compose up viz
```

SSH port forward:

```bash
ssh -L 8000:localhost:8000 jetson
```

Open [http://localhost:8000/viz/](http://localhost:8000/viz/)