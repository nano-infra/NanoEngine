import torch
import triton
import triton.language as tl


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


@triton.jit
def store_kcache_kernel(
    key_ptr,
    key_stride,
    k_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ids: pid_n over batch N, pid_d over D chunks

    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Load slot index for this batch element

    slot = tl.load(slot_mapping_ptr + pid_n)
    # If slot == -1, skip

    if slot == -1:
        return
    # Offsets into the D dimension for this chunk

    offs = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    # Compute flat offsets in key: [N, *, D] with stride key_stride along N

    key_offsets = pid_n * key_stride + offs
    key_vals = tl.load(key_ptr + key_offsets, mask=mask, other=0)

    # Compute flat offsets in cache: we assume slot-major, contiguous D region

    cache_offsets = slot * D + offs
    tl.store(k_cache_ptr + cache_offsets, key_vals, mask=mask)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


def store_kcache(
    key: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    # key: [N, num_heads, head_dim]

    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    # Layout checks (your existing asserts)
    assert key.stride(-1) == 1
    assert key.stride(1) == head_dim
    assert k_cache.stride(1) == D
    assert slot_mapping.numel() == N

    BLOCK_SIZE = 256  # power of 2; you can use 128/256/512 depending on perf/memory

    grid = (N, triton.cdiv(D, BLOCK_SIZE))
    store_kcache_kernel[grid](
        key, key.stride(0), k_cache, slot_mapping, D, BLOCK_SIZE=BLOCK_SIZE
    )
