import json
import math
from functools import lru_cache

import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool = True,
) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.float()
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    if is_neox_style:
        output = torch.cat((y1, y2), dim=-1)
    else:
        output = torch.stack((y1, y2), dim=-1).flatten(-2)
    return output.to(input_dtype)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> float:
    return (
        dim
        * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> tuple[int, int]:
    low = math.floor(
        yarn_find_correction_dim(
            low_rot, dim, base, max_position_embeddings
        )
    )
    high = math.ceil(
        yarn_find_correction_dim(
            high_rot, dim, base, max_position_embeddings
        )
    )
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(low: int, high: int, dim: int) -> torch.Tensor:
    if low == high:
        high += 0.001
    ramp = (torch.arange(dim, dtype=torch.float) - low) / (high - low)
    return torch.clamp(ramp, 0, 1)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        rope_scaling: dict | None = None,
        is_neox_style: bool = True,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.is_neox_style = is_neox_style
        assert rotary_dim == head_size
        pos_freqs = base ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim
        )
        inv_freq = 1.0 / pos_freqs
        magnitude_scale = 1.0

        if rope_scaling is not None:
            rope_type = rope_scaling.get(
                "rope_type", rope_scaling.get("type", "default")
            )
            if rope_type in {"yarn", "deepseek_yarn"}:
                scaling_factor = float(rope_scaling["factor"])
                original_max_position = int(
                    rope_scaling["original_max_position_embeddings"]
                )
                inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
                low, high = yarn_find_correction_range(
                    int(rope_scaling.get("beta_fast", 32)),
                    int(rope_scaling.get("beta_slow", 1)),
                    rotary_dim,
                    base,
                    original_max_position,
                )
                inv_freq_mask = (
                    1
                    - yarn_linear_ramp_mask(low, high, rotary_dim // 2)
                ) * float(rope_scaling.get("extrapolation_factor", 1))
                inv_freq = (
                    inv_freq_interpolation * (1 - inv_freq_mask)
                    + inv_freq * inv_freq_mask
                )
                magnitude_scale = (
                    yarn_get_mscale(
                        scaling_factor, float(rope_scaling.get("mscale", 1))
                    )
                    / yarn_get_mscale(
                        scaling_factor,
                        float(rope_scaling.get("mscale_all_dim", 0)),
                    )
                    * float(rope_scaling.get("attn_factor", 1))
                )

        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * magnitude_scale
        sin = freqs.sin() * magnitude_scale
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
        query_dtype = query.dtype
        key_dtype = key.dtype
        query = apply_rotary_emb(
            query, cos, sin, self.is_neox_style
        ).to(query_dtype)
        key = apply_rotary_emb(key, cos, sin, self.is_neox_style).to(key_dtype)
        return query, key


@lru_cache(1)
def _get_rope_cached(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling_json: str | None,
    is_neox_style: bool,
):
    rope_scaling = json.loads(rope_scaling_json) if rope_scaling_json else None
    return RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        rope_scaling=rope_scaling,
        is_neox_style=is_neox_style,
    )


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
    is_neox_style: bool = True,
):
    rope_scaling_json = (
        json.dumps(rope_scaling, sort_keys=True)
        if rope_scaling is not None
        else None
    )
    return _get_rope_cached(
        head_size,
        rotary_dim,
        max_position,
        base,
        rope_scaling_json,
        is_neox_style,
    )
