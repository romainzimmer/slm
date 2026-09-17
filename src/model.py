from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


def _round_up_multiple(value: int, multiple: int) -> int:
    return (-(value // -multiple)) * multiple


def swiglu_hidden_dim(dim: int, *, multiple: int = 256) -> int:
    return _round_up_multiple(round(8 * dim / 3), multiple)


PRESETS: dict[str, dict[str, int]] = {
    "tiny": {
        "dim": 256,
        "num_blocks": 2,
        "inner_iters": 4,
        "num_heads": 4,
        "num_kv_heads": 4,
        "max_seq_len": 1024,
    },
    "small": {
        "dim": 512,
        "num_blocks": 2,
        "inner_iters": 6,
        "num_heads": 8,
        "num_kv_heads": 4,
        "max_seq_len": 1024,
    },
    "medium": {
        "dim": 768,
        "num_blocks": 3,
        "inner_iters": 6,
        "num_heads": 12,
        "num_kv_heads": 4,
        "max_seq_len": 1024,
    },
}


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    dim: int = 256
    num_blocks: int = 2
    inner_iters: int = 4
    num_heads: int = 4
    num_kv_heads: int = 4
    max_seq_len: int = 1024
    seq_len: int = 512
    rope_theta: float = 10000.0
    loss_iters: int | None = None
    tie_weights: bool = True
    use_naive_attn: bool = False

    def __post_init__(self) -> None:
        if self.loss_iters is None:
            self.loss_iters = self.inner_iters
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.dim % self.num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

    @classmethod
    def from_preset(cls, preset: str, **overrides: object) -> ModelConfig:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        cfg = dict(PRESETS[preset])
        cfg.update(overrides)
        return cls(**cfg)


class ModelOutput(NamedTuple):
    logits: torch.Tensor
    iter_logits: list[torch.Tensor]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    *,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    )
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate Q/K. x: (B, H, T, D); cos/sin: (T, D) or (1, 1, T, D)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.to(x.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return x.unsqueeze(2).expand(b, n_kv, n_rep, t, d).reshape(b, n_kv * n_rep, t, d)


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    is_causal: bool = False,
) -> torch.Tensor:
    if mask is not None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.dim // cfg.num_heads
        self.n_rep = cfg.num_heads // cfg.num_kv_heads

        self.q_proj = nn.Linear(cfg.dim, cfg.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * self.head_dim, cfg.dim, bias=False)

    def _project(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        pos: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if pos is not None and t == 1:
            cos_t = cos[pos : pos + 1]
            sin_t = sin[pos : pos + 1]
        else:
            cos_t = cos[:t]
            sin_t = sin[:t]
        q = apply_rope(q, cos_t, sin_t)
        k = apply_rope(k, cos_t, sin_t)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None,
        naive_attn: bool,
        kv_cache: KVCache | None = None,
        loop_iter: int = 0,
        block_idx: int = 0,
        pos: int | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self._project(x, cos=cos, sin=sin, pos=pos)

        if kv_cache is not None:
            assert pos is not None
            k, v = kv_cache.update(loop_iter, block_idx, k, v, pos)

        if naive_attn:
            y = naive_attention(q, k, v, mask=mask)
        elif kv_cache is not None:
            y = sdpa_attention(q, k, v, mask=None, is_causal=False)
        else:
            y = sdpa_attention(q, k, v, mask=None, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hidden = swiglu_hidden_dim(dim)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg.dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None,
        naive_attn: bool,
        kv_cache: KVCache | None = None,
        loop_iter: int = 0,
        block_idx: int = 0,
        pos: int | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(
            h,
            cos=cos,
            sin=sin,
            mask=mask,
            naive_attn=naive_attn,
            kv_cache=kv_cache,
            loop_iter=loop_iter,
            block_idx=block_idx,
            pos=pos,
        )
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class KVCache:
    """Per-(loop_iter, block_idx) K/V tensors for autoregressive decode."""

    def __init__(self, max_seq_len: int) -> None:
        self.max_seq_len = max_seq_len
        self._cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def update(
        self,
        loop_iter: int,
        block_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (loop_iter, block_idx)
        if key not in self._cache:
            b, h, _, d = k.shape
            k_buf = k.new_zeros(b, h, self.max_seq_len, d)
            v_buf = v.new_zeros(b, h, self.max_seq_len, d)
            k_buf[:, :, pos : pos + 1, :] = k
            v_buf[:, :, pos : pos + 1, :] = v
            self._cache[key] = (k_buf, v_buf)
        else:
            k_buf, v_buf = self._cache[key]
            k_buf[:, :, pos : pos + 1, :] = k
            v_buf[:, :, pos : pos + 1, :] = v
        k_out, v_out = self._cache[key]
        return k_out[:, :, : pos + 1, :], v_out[:, :, : pos + 1, :]

    def clear(self) -> None:
        self._cache.clear()


class LoopedCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.num_blocks))
        self.norm = RMSNorm(cfg.dim)
        if cfg.tie_weights:
            self.lm_head: nn.Linear | None = None
        else:
            self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        head_dim = cfg.dim // cfg.num_heads
        cos, sin = build_rope_cache(
            cfg.max_seq_len,
            head_dim,
            theta=cfg.rope_theta,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer(
            "causal_mask",
            self._build_causal_mask(cfg.max_seq_len),
            persistent=False,
        )

    def _build_causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        return mask

    def get_lm_head_weight(self) -> torch.Tensor:
        if self.cfg.tie_weights:
            return self.embed.weight
        assert self.lm_head is not None
        return self.lm_head.weight

    def logits_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.get_lm_head_weight())

    def _apply_blocks(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None,
        naive_attn: bool,
        kv_cache: KVCache | None = None,
        loop_iter: int = 0,
        pos: int | None = None,
    ) -> torch.Tensor:
        for block_idx, block in enumerate(self.blocks):
            x = block(
                x,
                cos=cos,
                sin=sin,
                mask=mask,
                naive_attn=naive_attn,
                kv_cache=kv_cache,
                loop_iter=loop_iter,
                block_idx=block_idx,
                pos=pos,
            )
        return x

    def supervised_iters(
        self,
        inner_iters: int | None = None,
        *,
        loss_iters: int | None = None,
    ) -> range:
        b = inner_iters if inner_iters is not None else self.cfg.inner_iters
        t = min(loss_iters if loss_iters is not None else self.cfg.loss_iters, b)
        return range(b - t + 1, b + 1)

    def forward(
        self,
        idx: torch.Tensor,
        *,
        inner_iters: int | None = None,
        loss_iters: int | None = None,
        supervised_logits: bool | None = None,
        naive_attn: bool | None = None,
        kv_cache: KVCache | None = None,
        pos: int | None = None,
    ) -> ModelOutput:
        b, t = idx.shape
        naive = self.cfg.use_naive_attn if naive_attn is None else naive_attn
        iters = inner_iters if inner_iters is not None else self.cfg.inner_iters

        p = self.embed(idx)
        if kv_cache is not None:
            assert pos is not None
            p = p[:, -1:, :]
            t = 1

        cos = self.rope_cos.to(device=p.device)
        sin = self.rope_sin.to(device=p.device)
        mask = None if kv_cache is not None else self.causal_mask[:t, :t].to(device=p.device)

        h = torch.zeros_like(p)

        iter_logits: list[torch.Tensor] = []
        supervise = set(self.supervised_iters(iters, loss_iters=loss_iters))
        collect_logits = supervised_logits if supervised_logits is not None else self.training

        for loop_iter in range(1, iters + 1):
            inp = h + p
            h = self._apply_blocks(
                inp,
                cos=cos,
                sin=sin,
                mask=mask,
                naive_attn=naive,
                kv_cache=kv_cache,
                loop_iter=loop_iter - 1,
                pos=pos,
            )
            if loop_iter in supervise and collect_logits:
                iter_logits.append(self.logits_from_hidden(self.norm(h)))

        logits = self.logits_from_hidden(self.norm(h))
        return ModelOutput(logits=logits, iter_logits=iter_logits)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def compute_lm_loss(
    model: LoopedCausalLM,
    idx: torch.Tensor,
    targets: torch.Tensor,
    *,
    inner_iters: int | None = None,
    loss_iters: int | None = None,
    supervised_logits: bool | None = None,
) -> torch.Tensor:
    out = model(
        idx,
        inner_iters=inner_iters,
        loss_iters=loss_iters,
        supervised_logits=supervised_logits,
    )
    losses: list[torch.Tensor] = []
    if out.iter_logits:
        for logits in out.iter_logits:
            losses.append(
                F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            )
        return torch.stack(losses).mean()
    return F.cross_entropy(
        out.logits.reshape(-1, out.logits.size(-1)),
        targets.reshape(-1),
    )
