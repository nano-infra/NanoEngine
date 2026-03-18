"""Fused QKV-split + RMSNorm + RoPE Triton kernel for Ascend NPU.

Adapted from vllm-ascend (Apache 2.0):
  vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.py

Fuses 5+ ops per attention layer into a single kernel:
  1. Split packed QKV → Q, K, V
  2. Per-head RMSNorm on Q and K
  3. Rotary positional embedding on Q and K
  4. Write out Q, K, V

Saves ~189 RmsNorm + ~188 Slice + ~752 Mul + ~376 Cast ops per decode step
on Qwen3-MoE (94 layers).
"""

from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ---------------------------------------------------------------------------
# Ascend Triton utilities
# ---------------------------------------------------------------------------

_NUM_VECTORCORE = -1
_extension_module = None

if HAS_TRITON:
    try:
        import triton.language.extra.cann.extension as _extension_module
    except ImportError:
        _extension_module = None


def _resolve_triton_ascend_op(op_name: str):
    if _extension_module is not None:
        op = getattr(_extension_module, op_name, None)
        if op is not None:
            return op
    op = getattr(tl, op_name, None)
    if op is not None:
        return op
    raise RuntimeError(
        f"Failed to resolve Triton op '{op_name}': "
        "neither triton.language.extra.cann.extension nor triton.language provides it."
    )


if HAS_TRITON:
    insert_slice = _resolve_triton_ascend_op("insert_slice")
    extract_slice = _resolve_triton_ascend_op("extract_slice")
    get_element = _resolve_triton_ascend_op("get_element")


def _init_vectorcore_count():
    global _NUM_VECTORCORE
    if _NUM_VECTORCORE == -1 and HAS_TRITON:
        device_properties: dict[str, Any] = (
            triton.runtime.driver.active.utils.get_device_properties(
                torch.npu.current_device()
            )
        )
        _NUM_VECTORCORE = device_properties.get("num_vectorcore", -1)
        assert _NUM_VECTORCORE > 0, "Failed to detect vectorcore count."


def get_vectorcore_num() -> int:
    global _NUM_VECTORCORE
    if _NUM_VECTORCORE == -1:
        _init_vectorcore_count()
    return _NUM_VECTORCORE


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def split_qkv_rmsnorm_rope_kernel(
        input_gm_ptr,
        q_gm_ptr,
        k_gm_ptr,
        v_gm_ptr,
        q_weight_ptr,
        q_bias_ptr,
        k_weight_ptr,
        k_bias_ptr,
        batch_size,
        q_hidden_size: tl.constexpr,
        kv_hidden_size: tl.constexpr,
        total_hidden_size: tl.constexpr,
        eps: tl.constexpr,
        BIAS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROPE_DIM: tl.constexpr,
        HALF_ROPE_DIM: tl.constexpr,
        IS_PARTIAL_ROPE: tl.constexpr,
        num_vectorcore: tl.constexpr,
        batch_size_per_iter_per_vec: tl.constexpr,
        qk_head_nums_per_iter_per_vec: tl.constexpr,
        q_head_num: tl.constexpr,
        kv_head_num: tl.constexpr,
        qk_head_num_sum: tl.constexpr,
        v_batch_size_per_iter_per_vec: tl.constexpr,
        positions_gm_ptr,
        cos_sin_cache_gm_ptr,
    ):
        row_pid = tl.program_id(0)

        q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
        k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))

        batch_size_per_vec = tl.cdiv(batch_size, num_vectorcore)
        iter_num_per_vec = tl.cdiv(batch_size_per_vec, batch_size_per_iter_per_vec)
        v_iter_num_per_vec = tl.cdiv(
            batch_size_per_vec, v_batch_size_per_iter_per_vec
        )
        input_batch_offset = row_pid * batch_size_per_vec
        mblk_idx = tl.arange(0, batch_size_per_iter_per_vec) + input_batch_offset
        nblk_idx = tl.arange(0, q_hidden_size + kv_hidden_size)
        nmask = nblk_idx < total_hidden_size

        input_batch_offset_end = min(
            input_batch_offset + batch_size_per_vec, batch_size
        )

        pos_indices = input_batch_offset + tl.arange(
            0, batch_size_per_iter_per_vec
        )
        output_q_nblk_idx = tl.arange(0, q_hidden_size)
        output_q_nmask = output_q_nblk_idx < q_hidden_size
        output_kv_nblk_idx = tl.arange(0, kv_hidden_size)
        output_kv_nmask = output_kv_nblk_idx < kv_hidden_size
        sin_cos_range = tl.arange(0, ROPE_DIM)
        cos_sin_cache_offset = cos_sin_cache_gm_ptr + sin_cos_range

        for iter in tl.range(iter_num_per_vec):
            pos_offset = iter * batch_size_per_iter_per_vec
            x = tl.load(
                positions_gm_ptr + pos_indices + pos_offset,
                mask=(pos_indices + pos_offset) < input_batch_offset_end,
            )
            mmask = (mblk_idx + pos_offset) < input_batch_offset_end
            mask = (mmask[:, None]) & (nmask[None, :])
            idx = (mblk_idx + pos_offset)[:, None] * total_hidden_size + nblk_idx[
                None, :
            ]
            values_tmp1 = tl.load(input_gm_ptr + idx, mask=mask).reshape(
                qk_head_nums_per_iter_per_vec, HEAD_DIM
            )
            if BIAS:
                q_bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM))
                k_bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM))

            # --- Load cos/sin cache for each position ---
            values_tmp3 = tl.zeros(
                (batch_size_per_iter_per_vec, ROPE_DIM), dtype=tl.bfloat16
            )
            for i in tl.range(batch_size_per_iter_per_vec):
                pos = get_element(x, (i,))
                values_tmp3 = insert_slice(
                    values_tmp3.reshape(batch_size_per_iter_per_vec, ROPE_DIM),
                    tl.load(
                        pos * ROPE_DIM + cos_sin_cache_offset[:, None]
                    ).reshape(1, ROPE_DIM),
                    offsets=(i, 0),
                    sizes=(1, ROPE_DIM),
                    strides=(1, 1),
                )
            values_tmp3 = values_tmp3.reshape(
                batch_size_per_iter_per_vec, 1, ROPE_DIM
            )
            cos = extract_slice(
                values_tmp3,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM),
                strides=(1, 1, 1),
            )
            sin = extract_slice(
                values_tmp3,
                offsets=(0, 0, HALF_ROPE_DIM),
                sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM),
                strides=(1, 1, 1),
            )

            # --- Per-head RMSNorm on Q+K ---
            normalized_values = values_tmp1.to(tl.float32)
            normalized_values = normalized_values * normalized_values
            normalized_values = (
                tl.sum(normalized_values, axis=1) / HEAD_DIM
            )
            normalized_values = (
                1
                / tl.sqrt(normalized_values + eps).reshape(
                    qk_head_nums_per_iter_per_vec, 1
                )
            )
            normalized_values = values_tmp1 * normalized_values

            # --- Q: extract, weight, RoPE ---
            normalized_values_tmp = extract_slice(
                normalized_values.reshape(
                    batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM
                ),
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, q_head_num, HEAD_DIM),
                strides=(1, 1, 1),
            )

            if BIAS:
                normalized_values_tmp = (
                    normalized_values_tmp * q_weight_values + q_bias_values
                ).to(tl.bfloat16)
            else:
                normalized_values_tmp = (
                    normalized_values_tmp * q_weight_values
                ).to(tl.bfloat16)

            # Q RoPE
            values_tmp = tl.zeros(
                (batch_size_per_iter_per_vec, q_head_num, ROPE_DIM),
                dtype=tl.bfloat16,
            )
            x1 = extract_slice(
                normalized_values_tmp,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
                strides=(1, 1, 1),
            )
            x2 = extract_slice(
                normalized_values_tmp,
                offsets=(0, 0, HALF_ROPE_DIM),
                sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
                strides=(1, 1, 1),
            )
            values_tmp = insert_slice(
                values_tmp,
                x1 * cos - x2 * sin,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
                strides=(1, 1, 1),
            )
            values_tmp = insert_slice(
                values_tmp,
                x2 * cos + x1 * sin,
                offsets=(0, 0, HALF_ROPE_DIM),
                sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
                strides=(1, 1, 1),
            )
            q_output_idx = (
                output_q_nblk_idx[None, :]
                + (mblk_idx + pos_offset)[:, None] * q_hidden_size
            )
            mask = (mmask[:, None]) & (output_q_nmask[None, :])
            if IS_PARTIAL_ROPE:
                normalized_values_tmp = insert_slice(
                    normalized_values_tmp,
                    values_tmp,
                    offsets=(0, 0, 0),
                    sizes=(
                        batch_size_per_iter_per_vec,
                        q_head_num,
                        ROPE_DIM,
                    ),
                    strides=(1, 1, 1),
                )
                tl.store(
                    q_gm_ptr + q_output_idx,
                    normalized_values_tmp.reshape(
                        batch_size_per_iter_per_vec, q_hidden_size
                    ),
                    mask=mask,
                )
            else:
                tl.store(
                    q_gm_ptr + q_output_idx,
                    values_tmp.reshape(
                        batch_size_per_iter_per_vec, q_hidden_size
                    ),
                    mask=mask,
                )

            # --- K: extract, weight, RoPE ---
            normalized_values_tmp1 = extract_slice(
                normalized_values.reshape(
                    batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM
                ),
                offsets=(0, q_head_num, 0),
                sizes=(batch_size_per_iter_per_vec, kv_head_num, HEAD_DIM),
                strides=(1, 1, 1),
            )

            if BIAS:
                normalized_values_tmp1 = (
                    normalized_values_tmp1 * k_weight_values + k_bias_values
                ).to(tl.bfloat16)
            else:
                normalized_values_tmp1 = (
                    normalized_values_tmp1 * k_weight_values
                ).to(tl.bfloat16)

            values_tmp2 = tl.zeros(
                (batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM),
                dtype=tl.bfloat16,
            )
            x1 = extract_slice(
                normalized_values_tmp1,
                offsets=(0, 0, 0),
                sizes=(
                    batch_size_per_iter_per_vec,
                    kv_head_num,
                    HALF_ROPE_DIM,
                ),
                strides=(1, 1, 1),
            )
            x2 = extract_slice(
                normalized_values_tmp1,
                offsets=(0, 0, HALF_ROPE_DIM),
                sizes=(
                    batch_size_per_iter_per_vec,
                    kv_head_num,
                    HALF_ROPE_DIM,
                ),
                strides=(1, 1, 1),
            )
            values_tmp2 = insert_slice(
                values_tmp2,
                x1 * cos - x2 * sin,
                offsets=(0, 0, 0),
                sizes=(
                    batch_size_per_iter_per_vec,
                    kv_head_num,
                    HALF_ROPE_DIM,
                ),
                strides=(1, 1, 1),
            )
            values_tmp2 = insert_slice(
                values_tmp2,
                x2 * cos + x1 * sin,
                offsets=(0, 0, HALF_ROPE_DIM),
                sizes=(
                    batch_size_per_iter_per_vec,
                    kv_head_num,
                    HALF_ROPE_DIM,
                ),
                strides=(1, 1, 1),
            )

            kv_output_idx = (
                output_kv_nblk_idx[None, :]
                + (mblk_idx + pos_offset)[:, None] * kv_hidden_size
            )
            mask = (mmask[:, None]) & (output_kv_nmask[None, :])
            if IS_PARTIAL_ROPE:
                normalized_values_tmp1 = insert_slice(
                    normalized_values_tmp1,
                    values_tmp2,
                    offsets=(0, 0, 0),
                    sizes=(
                        batch_size_per_iter_per_vec,
                        kv_head_num,
                        ROPE_DIM,
                    ),
                    strides=(1, 1, 1),
                )
                tl.store(
                    k_gm_ptr + kv_output_idx,
                    normalized_values_tmp1.reshape(
                        batch_size_per_iter_per_vec, kv_hidden_size
                    ),
                    mask=mask,
                )
            else:
                tl.store(
                    k_gm_ptr + kv_output_idx,
                    values_tmp2.reshape(
                        batch_size_per_iter_per_vec, kv_hidden_size
                    ),
                    mask=mask,
                )

        # --- V: simple copy ---
        mblk_idx = (
            tl.arange(0, v_batch_size_per_iter_per_vec) + input_batch_offset
        )
        nblk_idx = tl.arange(
            q_hidden_size + kv_hidden_size, total_hidden_size
        )
        nmask = nblk_idx < total_hidden_size
        out_nblk_idx = tl.arange(0, kv_hidden_size)
        out_nmask = out_nblk_idx < kv_hidden_size

        for _ in tl.range(v_iter_num_per_vec):
            mmask = mblk_idx < input_batch_offset_end
            mask = (mmask[:, None]) & (nmask[None, :])
            idx = mblk_idx[:, None] * total_hidden_size + nblk_idx[None, :]
            values = tl.load(input_gm_ptr + idx, mask=mask)
            out_idx = (
                mblk_idx[:, None] * kv_hidden_size + out_nblk_idx[None, :]
            )
            out_mask = (mmask[:, None]) & (out_nmask[None, :])
            tl.store(v_gm_ptr + out_idx, values, mask=out_mask)
            mblk_idx += v_batch_size_per_iter_per_vec


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def split_qkv_rmsnorm_rope(
    qkv_input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QKV split + per-head RMSNorm + RoPE.

    Args:
        qkv_input: [batch, q_hidden + kv_hidden * 2] packed QKV projection output.
        cos_sin_cache: [max_position, rope_dim] cos/sin cache (concatenated).
        positions: [batch] position indices (int64).
        q_weight: [head_dim] RMSNorm weight for Q heads.
        k_weight: [head_dim] RMSNorm weight for K heads.
        q_hidden_size: Total Q hidden size (num_q_heads * head_dim).
        kv_hidden_size: Total KV hidden size (num_kv_heads * head_dim).
        head_dim: Dimension per head.
        eps: RMSNorm epsilon.
        q_bias: Optional [head_dim] RMSNorm bias for Q.
        k_bias: Optional [head_dim] RMSNorm bias for K.

    Returns:
        (q, k, v) each as flat 2D tensors:
          q: [batch, q_hidden_size]
          k: [batch, kv_hidden_size]
          v: [batch, kv_hidden_size]
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for fused QKV+RMSNorm+RoPE kernel")

    num_vectorcore = get_vectorcore_num()
    rope_dim = cos_sin_cache.shape[-1]
    batch_size = qkv_input.shape[0]
    BIAS = q_bias is not None
    IS_PARTIAL_ROPE = rope_dim != head_dim
    total_hidden_size = q_hidden_size + kv_hidden_size * 2

    q_output = torch.empty(
        batch_size, q_hidden_size, device=qkv_input.device, dtype=qkv_input.dtype
    )
    k_output = torch.empty(
        batch_size, kv_hidden_size, device=qkv_input.device, dtype=qkv_input.dtype
    )
    v_output = torch.empty(
        batch_size, kv_hidden_size, device=qkv_input.device, dtype=qkv_input.dtype
    )

    q_head_num = q_hidden_size // head_dim
    kv_head_num = kv_hidden_size // head_dim

    # UB (Unified Buffer) tiling calculation for Ascend vector core
    UB_SIZE = 87040  # 85K = 85 * 1024
    if IS_PARTIAL_ROPE:
        factor = (
            5 * q_hidden_size
            + 3 * kv_hidden_size
            + rope_dim * 4
            + q_head_num * rope_dim
        )
    else:
        factor = (
            5 * q_hidden_size
            + 3 * kv_hidden_size
            + rope_dim * 2
            + q_head_num * rope_dim // 2
        )
    batch_size_per_iter_per_vec = int(UB_SIZE / qkv_input.element_size()) // factor
    batch_size_per_iter_per_vec = max(1, batch_size_per_iter_per_vec)
    qk_head_num_sum = int(q_head_num + kv_head_num)
    qk_head_nums_per_iter_per_vec = batch_size_per_iter_per_vec * qk_head_num_sum

    grid = (num_vectorcore, 1, 1)
    v_batch_size_per_iter_per_vec = (
        UB_SIZE / torch.bfloat16.itemsize // (kv_hidden_size + 1)
    )

    split_qkv_rmsnorm_rope_kernel[grid](
        qkv_input,
        q_output,
        k_output,
        v_output,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        batch_size,
        q_hidden_size,
        kv_hidden_size,
        total_hidden_size,
        eps,
        BIAS,
        head_dim,
        rope_dim,
        rope_dim // 2,
        IS_PARTIAL_ROPE,
        num_vectorcore,
        int(batch_size_per_iter_per_vec),
        int(qk_head_nums_per_iter_per_vec),
        q_head_num,
        kv_head_num,
        qk_head_num_sum,
        int(v_batch_size_per_iter_per_vec),
        positions,
        cos_sin_cache,
    )
    return q_output, k_output, v_output
