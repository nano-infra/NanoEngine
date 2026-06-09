"""Triton kernels for CUDA-Graph-safe local MoE token dispatch & combine.

These kernels implement a padded routing scheme (EP==1 / TP decode path):

  dispatch: scatter hidden[T, H] → padded_buf[E, max_m, H]  (atomicAdd slot)
  combine:  gather  expert_out[E, max_m, H] → out[T, H]     (weighted sum)

The output layout [E, max_m, H] + masked_m[E] matches what DeepEP
``low_latency_dispatch`` produces, allowing the same masked grouped GEMMs
to be used for both EP==1 and EP>1 on Hopper.
"""

import triton
import triton.language as tl

ALIGNMENT = 128  # DeepGEMM block-size alignment – max_m rounded to this


# ---------------------------------------------------------------------------
# Dispatch kernel  (scatter hidden → padded_buf)
# ---------------------------------------------------------------------------


@triton.jit
def local_dispatch_kernel(
    hidden_ptr,  # [T, H]  bf16
    topk_ids_ptr,  # [T, K]  int64
    padded_buf_ptr,  # [E, max_m, H]  bf16  (zero-inited before launch)
    masked_m_ptr,  # [E]  int32          (zero-inited before launch)
    slot_map_ptr,  # [T, K]  int32        (output: slot index; -1 if dropped)
    num_experts,  # E  – runtime scalar
    max_m,  # per-expert capacity – runtime scalar
    H,  # hidden dimension    – runtime scalar
    stride_h_t,
    stride_h_h,
    stride_topk_t,
    stride_topk_k,
    stride_pb_e,
    stride_pb_m,
    stride_pb_h,
    stride_sm_t,
    stride_sm_k,
    BLOCK_H: tl.constexpr,
    TOP_K: tl.constexpr,
):
    """One program per (token, top_k) pair."""
    pid = tl.program_id(0)
    t = pid // TOP_K
    k = pid % TOP_K

    e = tl.load(topk_ids_ptr + t * stride_topk_t + k * stride_topk_k)

    # Invalid expert → mark dropped and bail
    if e < 0 or e >= num_experts:
        tl.store(slot_map_ptr + t * stride_sm_t + k * stride_sm_k, -1)
        return

    # Atomically claim a slot within this expert's column
    slot = tl.atomic_add(masked_m_ptr + e, 1)

    if slot >= max_m:
        # Capacity overflow → drop
        tl.store(slot_map_ptr + t * stride_sm_t + k * stride_sm_k, -1)
        return

    tl.store(slot_map_ptr + t * stride_sm_t + k * stride_sm_k, slot)

    # Vectorised copy: hidden[t, :H] → padded_buf[e, slot, :H]
    src_base = t * stride_h_t
    dst_base = e * stride_pb_e + slot * stride_pb_m
    for h_start in tl.range(0, H, BLOCK_H):
        h_off = h_start + tl.arange(0, BLOCK_H)
        mask = h_off < H
        vals = tl.load(hidden_ptr + src_base + h_off * stride_h_h, mask=mask)
        tl.store(padded_buf_ptr + dst_base + h_off * stride_pb_h, vals, mask=mask)


# ---------------------------------------------------------------------------
# Combine kernel  (gather expert_out → weighted sum per token)
# ---------------------------------------------------------------------------


@triton.jit
def local_combine_kernel(
    expert_out_ptr,  # [E, max_m, H_out]
    topk_ids_ptr,  # [T, K]  int64
    topk_weights_ptr,  # [T, K]  float32 or bf16
    slot_map_ptr,  # [T, K]  int32
    out_ptr,  # [T, H_out]  bf16  (zero-inited before launch)
    H_out,
    stride_eo_e,
    stride_eo_m,
    stride_eo_h,
    stride_topk_t,
    stride_topk_k,
    stride_tw_t,
    stride_tw_k,
    stride_sm_t,
    stride_sm_k,
    stride_out_t,
    stride_out_h,
    BLOCK_H: tl.constexpr,
    TOP_K: tl.constexpr,
):
    """Grid: (T, num_h_blocks).  Each program reduces K expert contributions."""
    t = tl.program_id(0)
    h_block = tl.program_id(1)

    h_off = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h_off < H_out

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for k in tl.static_range(0, TOP_K):
        slot = tl.load(slot_map_ptr + t * stride_sm_t + k * stride_sm_k)
        if slot >= 0:
            e = tl.load(topk_ids_ptr + t * stride_topk_t + k * stride_topk_k)
            w = tl.load(topk_weights_ptr + t * stride_tw_t + k * stride_tw_k).to(
                tl.float32
            )
            val = tl.load(
                expert_out_ptr
                + e * stride_eo_e
                + slot * stride_eo_m
                + h_off * stride_eo_h,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += w * val

    tl.store(
        out_ptr + t * stride_out_t + h_off * stride_out_h,
        acc.to(tl.bfloat16),
        mask=mask,
    )
