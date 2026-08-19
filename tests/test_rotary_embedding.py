import math

import torch

from nanodeploy.layers.rotary_embedding import apply_rotary_emb, get_rope


def _reference_yarn_inv_freq(
    rotary_dim: int,
    base: float,
    scaling: dict,
) -> torch.Tensor:
    pos_freqs = base ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim
    )
    extrapolated = 1.0 / pos_freqs
    interpolated = extrapolated / scaling["factor"]

    def correction_dim(num_rotations: int) -> float:
        return (
            rotary_dim
            * math.log(
                scaling["original_max_position_embeddings"]
                / (num_rotations * 2 * math.pi)
            )
            / (2 * math.log(base))
        )

    low = max(math.floor(correction_dim(scaling["beta_fast"])), 0)
    high = min(math.ceil(correction_dim(scaling["beta_slow"])), rotary_dim - 1)
    ramp = torch.clamp(
        (torch.arange(rotary_dim // 2, dtype=torch.float) - low)
        / (high - low),
        0,
        1,
    )
    extrapolation_mask = 1 - ramp
    return (
        interpolated * (1 - extrapolation_mask)
        + extrapolated * extrapolation_mask
    )


def test_deepseek_yarn_rope_uses_interleaved_pairs_and_scaled_frequencies():
    rotary_dim = 8
    base = 10_000.0
    scaling = {
        "type": "yarn",
        "factor": 4,
        "original_max_position_embeddings": 64,
        "beta_fast": 2,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    }
    rope = get_rope(
        rotary_dim,
        rotary_dim,
        max_position=256,
        base=base,
        rope_scaling=scaling,
        is_neox_style=False,
    )

    position = 37
    inv_freq = _reference_yarn_inv_freq(rotary_dim, base, scaling)
    expected_cos = (position * inv_freq).cos()
    expected_sin = (position * inv_freq).sin()
    actual_cos, actual_sin = rope.cos_sin_cache[position, 0].chunk(2)
    torch.testing.assert_close(actual_cos, expected_cos)
    torch.testing.assert_close(actual_sin, expected_sin)

    query = torch.arange(1, rotary_dim + 1, dtype=torch.float).view(1, 1, -1)
    expected_query = torch.stack(
        (
            query[..., ::2] * expected_cos - query[..., 1::2] * expected_sin,
            query[..., 1::2] * expected_cos + query[..., ::2] * expected_sin,
        ),
        dim=-1,
    ).flatten(-2)
    actual_query, actual_key = rope.forward.__wrapped__(
        rope,
        torch.tensor([position]),
        query,
        query.clone(),
    )
    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_key, expected_query)


def test_default_rope_keeps_neox_half_pairing():
    x = torch.arange(1, 9, dtype=torch.float).view(1, 1, -1)
    cos = torch.zeros(1, 1, 4)
    sin = torch.ones(1, 1, 4)

    actual = apply_rotary_emb(x, cos, sin)
    expected = torch.tensor([[[-5.0, -6.0, -7.0, -8.0, 1.0, 2.0, 3.0, 4.0]]])
    torch.testing.assert_close(actual, expected)
