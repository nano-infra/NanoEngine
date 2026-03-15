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
    valid_mask = slot_mapping != -1
    if not valid_mask.any():
        return

    slots = slot_mapping[valid_mask]
    # Flatten k_cache to [num_slots, num_kv_heads, head_dim] for index assignment
    k_flat = k_cache.view(-1, k_cache.shape[-2], k_cache.shape[-1])
    v_flat = v_cache.view(-1, v_cache.shape[-2], v_cache.shape[-1])
    k_flat[slots] = key[valid_mask]
    v_flat[slots] = value[valid_mask]


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
    valid_mask = slot_mapping != -1
    if not valid_mask.any():
        return

    slots = slot_mapping[valid_mask]
    k_flat = k_cache.view(-1, k_cache.shape[-1])
    k_flat[slots] = key[valid_mask]
