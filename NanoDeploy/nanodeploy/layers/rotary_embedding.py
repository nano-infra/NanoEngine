import math
from functools import lru_cache

import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


# --- Yarn RoPE helpers ---


def _yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    low = math.floor(max(low, 0))
    high = math.ceil(min(high, dim - 1))
    return low, high


def _yarn_linear_ramp_mask(low, high, dim):
    if low == high:
        high += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
    return torch.clamp(linear_func, 0, 1)


class YarnRotaryEmbedding(nn.Module):
    """Yarn-aware rotary embedding (same forward interface as RotaryEmbedding)."""

    def __init__(
        self,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
    ) -> None:
        super().__init__()
        self.rotary_dim = rotary_dim
        self.scaling_factor = scaling_factor

        # Compute yarn-interpolated inv_freq
        freq_extra = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        freq_inter = 1.0 / (
            scaling_factor
            * base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, rotary_dim, base, original_max_position_embeddings
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2).to(
            dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

        # Build cos/sin cache
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()

        # Apply mscale
        emb_mscale = _yarn_get_mscale(scaling_factor, mscale) / _yarn_get_mscale(
            scaling_factor, mscale_all_dim
        )
        if emb_mscale != 1.0:
            cos = cos * emb_mscale
            sin = sin * emb_mscale

        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


def _yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


@lru_cache(1)
def _get_rope_cached(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling_hash: tuple | None = None,
):
    # Convert hashable tuple back to dict if needed
    rope_scaling = dict(rope_scaling_hash) if rope_scaling_hash else None

    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
        if rope_type == "yarn":
            rotary_emb = YarnRotaryEmbedding(
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position,
                base=base,
                scaling_factor=rope_scaling.get("factor", 1.0),
                original_max_position_embeddings=rope_scaling.get(
                    "original_max_position_embeddings", 4096
                ),
                beta_fast=rope_scaling.get("beta_fast", 32.0),
                beta_slow=rope_scaling.get("beta_slow", 1.0),
                mscale=rope_scaling.get("mscale", 1.0),
                mscale_all_dim=rope_scaling.get("mscale_all_dim", 0.0),
            )
        else:
            rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    else:
        rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | tuple | None = None,
):
    # Convert dict to hashable tuple for caching
    if rope_scaling is None:
        rope_scaling_hash = None
    elif isinstance(rope_scaling, dict):
        rope_scaling_hash = tuple(sorted(rope_scaling.items()))
    else:
        rope_scaling_hash = rope_scaling

    return _get_rope_cached(
        head_size, rotary_dim, max_position, base, rope_scaling_hash
    )
