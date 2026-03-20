"""Ascend NPU KV cache store operations.

Replaces the Triton kernels used on CUDA with pure PyTorch index ops that
run correctly on Ascend NPU (no Triton support on Ascend).
"""

import torch


def store_kvcache_npu(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Store new K/V tokens into paged KV cache slots.

    Args:
        key:          [N, num_heads, head_dim] — new key tokens
        value:        [N, num_heads, head_dim] — new value tokens
        k_cache:      [num_blocks*block_size, num_heads, head_dim] (flattened view)
        v_cache:      [num_blocks*block_size, num_heads, head_dim] (flattened view)
        slot_mapping: [N] int64 — target flat slot index per token (-1 = skip)
    """
    k_flat = k_cache.view(-1, k_cache.shape[-2], k_cache.shape[-1])
    v_flat = v_cache.view(-1, v_cache.shape[-2], v_cache.shape[-1])
    valid = slot_mapping >= 0                              # [N] bool, stays on NPU
    # Route invalid (-1) slots to the last cache slot as a harmless dummy target.
    # Decode (graph capture/replay) always has all-valid slots; prefill runs eager.
    dummy_idx = k_flat.shape[0] - 1                       # Python int — compile-time const
    safe_slots = torch.where(valid, slot_mapping,
                             slot_mapping.new_full((), dummy_idx))  # [N], no D2H
    vm = valid.view(-1, 1, 1)                              # [N, 1, 1]
    # For invalid slots: gather existing cache value and write it back (true no-op).
    # Output shape [N, num_heads, head_dim] is statically determined — ACL-graph safe.
    k_flat.index_put_((safe_slots,), torch.where(vm, key, k_flat[safe_slots]),
                      accumulate=False)
    v_flat.index_put_((safe_slots,), torch.where(vm, value, v_flat[safe_slots]),
                      accumulate=False)


def store_kcache_npu(
    key: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Store new key tokens into a paged K cache (MLA / single-cache mode).

    Args:
        key:          [N, head_dim] — new key tokens (already compressed)
        k_cache:      [num_blocks*block_size, head_dim] (flattened view)
        slot_mapping: [N] int64 — target flat slot index per token (-1 = skip)
    """
    k_flat = k_cache.view(-1, k_cache.shape[-1])
    valid = slot_mapping >= 0                              # [N] bool
    dummy_idx = k_flat.shape[0] - 1
    safe_slots = torch.where(valid, slot_mapping,
                             slot_mapping.new_full((), dummy_idx))  # [N]
    vm = valid.view(-1, 1)                                 # [N, 1]
    k_flat.index_put_((safe_slots,), torch.where(vm, key, k_flat[safe_slots]),
                      accumulate=False)
