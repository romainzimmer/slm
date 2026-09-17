# Run directories

```
runs/<run-id>/
  history.json
  manifest.json
  last.pt
  best.pt
  epochs/0003.pt
  samples/epoch_0003.json
```

## history.json

- `args`: full training config
- `epochs[]`: per-epoch train/val loss, ppl, bpc, lr_muon, lr_adam

## manifest.json

Lists sample epochs available under `samples/`.

## Viz

After training, from `jetson/`:

```bash
docker compose up viz
```

Open [http://localhost:8000/viz/](http://localhost:8000/viz/)

The UI shows:

- Train / validation metric charts per epoch
- Generated text samples at each saved epoch

Train and validation metrics per epoch: loss, perplexity, BPC, and learning rates.

<p align="center">
  <img src="assets/charts-viz.png" width="560" alt="Training metrics">
</p>

Generated text samples at each saved epoch.

<p align="center">
  <img src="assets/samples-viz.png" width="560" alt="Generated samples">
</p>
