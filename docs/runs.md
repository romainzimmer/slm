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
