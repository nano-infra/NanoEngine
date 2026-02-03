#!/usr/bin/env python3
"""
Unit test to verify KV cache tensor shape and RDMA read operation.
This test simulates the RDMA read operation on a KV cache tensor to verify
that the offset calculation correctly maps to the tensor's physical memory layout.
"""

import numpy as np
import torch


def test_kvcache_tensor_layout():
    """Test KV cache tensor layout and offset calculation."""

    # Parameters from logs
    block_size = 256
    num_heads = 4
    head_dim = 128
    dtype = torch.float16
    dtype_size = 2  # bytes for float16

    # KV cache shape: [kv_count, num_layers, num_blocks, block_size, num_heads, head_dim]
    kv_count = 2
    num_layers = 94
    num_blocks = 13  # blocks 0-12, but valid blocks start from 1

    print("=" * 80)
    print("KV Cache Tensor Layout Test")
    print("=" * 80)

    # Create a test KV cache tensor
    kv_cache = torch.randn(
        kv_count, num_layers, num_blocks, block_size, num_heads, head_dim, dtype=dtype
    )

    print(f"\n1. KV Cache Tensor Shape: {kv_cache.shape}")
    print(f"   Total size: {kv_cache.numel()} elements")
    print(f"   Total bytes: {kv_cache.numel() * dtype_size} bytes")
    print(f"   Memory layout: {kv_cache.is_contiguous()}")

    # Calculate block stride
    block_stride = block_size * num_heads * head_dim * dtype_size
    print(f"\n2. Block Stride: {block_stride} bytes")

    # Test offset calculation for a specific block
    kv_idx = 0
    layer_idx = 0
    block_idx = 1  # First valid block (0 is dummy)

    print(f"\n3. Testing offset calculation for:")
    print(f"   kv_idx={kv_idx}, layer_idx={layer_idx}, block_idx={block_idx}")

    # Calculate offset using the same logic as in cache.py
    def block_stride_func(block_idx):
        return block_idx * block_size * num_heads * head_dim * dtype_size

    def local_layer_stride(layer_idx, block_idx):
        return block_stride_func(num_blocks) * layer_idx + block_stride_func(block_idx)

    def local_kv_stride(kv_idx, layer_idx, block_idx):
        return local_layer_stride(num_layers, 0) * kv_idx + local_layer_stride(
            layer_idx, block_idx
        )

    calculated_offset = local_kv_stride(kv_idx, layer_idx, block_idx)
    print(f"   Calculated offset: {calculated_offset} bytes")

    # Get the actual tensor slice
    tensor_slice = kv_cache[kv_idx, layer_idx, block_idx, :, :, :]
    print(f"   Tensor slice shape: {tensor_slice.shape}")
    print(f"   Tensor slice size: {tensor_slice.numel()} elements")
    print(f"   Tensor slice bytes: {tensor_slice.numel() * dtype_size} bytes")

    # Get the actual memory offset of the tensor slice
    tensor_data_ptr = kv_cache.data_ptr()
    slice_data_ptr = tensor_slice.data_ptr()
    actual_offset = slice_data_ptr - tensor_data_ptr

    print(f"\n4. Memory Layout Verification:")
    print(f"   KV cache base pointer: {tensor_data_ptr}")
    print(f"   Slice pointer: {slice_data_ptr}")
    print(f"   Actual memory offset: {actual_offset} bytes")
    print(f"   Calculated offset: {calculated_offset} bytes")
    print(f"   Match: {'✓' if actual_offset == calculated_offset else '✗'}")

    if actual_offset != calculated_offset:
        print(
            f"   ERROR: Offset mismatch! Difference: {abs(actual_offset - calculated_offset)} bytes"
        )
        return False

    # Test reading data using byte-level access
    print(f"\n5. Testing byte-level read simulation:")

    # Create a flat view of the tensor
    flat_tensor = kv_cache.flatten()
    flat_bytes = flat_tensor.view(torch.uint8)

    # Simulate RDMA read: read from offset to offset+length
    read_offset = calculated_offset
    read_length = block_stride_func(1)  # One block
    read_end = read_offset + read_length

    print(
        f"   Reading from byte offset {read_offset} to {read_end} (length={read_length})"
    )

    if read_end > len(flat_bytes):
        print(f"   ERROR: Read range exceeds tensor size!")
        return False

    # Read the bytes
    read_bytes = flat_bytes[read_offset:read_end]

    # Convert back to tensor
    read_tensor = read_bytes.view(dtype).reshape(block_size, num_heads, head_dim)

    print(f"   Read tensor shape: {read_tensor.shape}")
    print(f"   Original slice shape: {tensor_slice.shape}")

    # Compare
    if torch.allclose(read_tensor, tensor_slice, atol=1e-5):
        print(f"   ✓ Read data matches original tensor slice!")
    else:
        print(f"   ✗ Read data does NOT match original tensor slice!")
        print(
            f"   Max difference: {torch.max(torch.abs(read_tensor - tensor_slice)).item()}"
        )
        return False

    # Test multiple layers
    print(f"\n6. Testing multiple layers:")
    for layer_idx in [0, 1, 2]:
        offset = local_kv_stride(kv_idx, layer_idx, block_idx)
        slice_data = kv_cache[kv_idx, layer_idx, block_idx, :, :, :]
        slice_ptr = slice_data.data_ptr()
        actual_off = slice_ptr - tensor_data_ptr

        print(
            f"   Layer {layer_idx}: offset={offset}, actual={actual_off}, match={'✓' if offset == actual_off else '✗'}"
        )
        if offset != actual_off:
            return False

    print(f"\n{'=' * 80}")
    print("✓ All tests passed!")
    print(f"{'=' * 80}")
    return True


def test_rdma_read_format():
    """Test RDMA read format with actual tensor data."""

    print("\n" + "=" * 80)
    print("RDMA Read Format Test")
    print("=" * 80)

    # Create source and destination tensors
    block_size = 256
    num_heads = 4
    head_dim = 128
    dtype = torch.float16

    # Source tensor (remote, prefill engine)
    source_tensor = torch.randn(block_size, num_heads, head_dim, dtype=dtype)
    source_flat = source_tensor.flatten().view(torch.uint8)

    # Destination tensor (local, decode engine) - initialized to zeros
    dest_tensor = torch.zeros(block_size, num_heads, head_dim, dtype=dtype)
    dest_flat = dest_tensor.flatten().view(torch.uint8)

    print(f"\n1. Source tensor shape: {source_tensor.shape}")
    print(f"   Source tensor mean: {source_tensor.mean().item():.6f}")
    print(f"   Destination tensor shape: {dest_tensor.shape}")
    print(f"   Destination tensor mean: {dest_tensor.mean().item():.6f}")

    # Simulate RDMA read: (local_handler, remote_handler, remote_offset, local_offset, length)
    # Format: read from remote[remote_off:remote_off+length] to local[local_off:local_off+length]
    remote_off = 0  # Start of source tensor
    local_off = 0  # Start of destination tensor
    length = source_flat.numel()  # Full block size in bytes

    print(f"\n2. Simulating RDMA read:")
    print(
        f"   Format: (local_handler, remote_handler, remote_offset={remote_off}, local_offset={local_off}, length={length})"
    )
    print(
        f"   Meaning: read from remote[{remote_off}:{remote_off+length}] to local[{local_off}:{local_off+length}]"
    )

    # Perform the "read" operation
    dest_flat[local_off : local_off + length] = source_flat[
        remote_off : remote_off + length
    ]

    # Convert back to tensor
    dest_tensor_after = dest_flat.view(dtype).reshape(block_size, num_heads, head_dim)

    print(f"\n3. After RDMA read:")
    print(f"   Destination tensor mean: {dest_tensor_after.mean().item():.6f}")

    # Verify
    if torch.allclose(dest_tensor_after, source_tensor, atol=1e-5):
        print(f"   ✓ RDMA read successful! Data matches source.")
        return True
    else:
        print(f"   ✗ RDMA read failed! Data does not match source.")
        print(
            f"   Max difference: {torch.max(torch.abs(dest_tensor_after - source_tensor)).item()}"
        )
        return False


if __name__ == "__main__":
    print("Running KV Cache RDMA Read Unit Tests\n")

    test1_passed = test_kvcache_tensor_layout()
    test2_passed = test_rdma_read_format()

    if test1_passed and test2_passed:
        print("\n✓ All unit tests passed!")
        exit(0)
    else:
        print("\n✗ Some unit tests failed!")
        exit(1)
