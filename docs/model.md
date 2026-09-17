# Model

Looped causal LM: $Y_0 = 0$, $Y_t = M(Y_{t-1} + P)$ for $t = 1..b$.

- **M**: shared stack of `num_blocks` decoder layers (RoPE, RMSNorm, GQA, SwiGLU)
- **P**: token embeddings (re-added each loop)
- **Loss**: CE on last `loss_iters` loop outputs (deep supervision)

## Presets

| Name | dim | blocks | inner_iters | ~params |
|------|-----|--------|-------------|---------|
| tiny | 256 | 2 | 4 | ~8M |
| small | 512 | 2 | 6 | ~28M |
| medium | 768 | 3 | 6 | ~75M |

## KV cache (v1)

Cache key: `(loop_iter, block_idx)`. Each inner loop pass stores separate K/V activations per block. One token per decode step; cache grows by position index.

`generate.py` uses incremental KV cache for all loop counts (separate K/V per inner loop × block). Tests verify cached decode matches full-prefix forward for `inner_iters` 1, 2, and 4.

## Attention

- Default: `F.scaled_dot_product_attention`
- `--naive-attn`: explicit softmax path for tests
