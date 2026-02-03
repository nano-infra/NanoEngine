#!/usr/bin/env python3
"""
Unit test to verify RDMA read operation using peer_agent.
This test creates two peer agents, registers KV cache memory regions,
and performs RDMA read operations to verify data correctness.
"""

import os
import sys
import time

import numpy as np
import torch

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DLSlime"))
from dlslime import start_peer_agent


def test_peer_agent_rdma_read():
    """Test RDMA read using peer_agent with KV cache tensor."""

    print("=" * 80)
    print("Peer Agent RDMA Read Test")
    print("=" * 80)

    # Parameters matching actual KV cache (from logs)
    block_size = 256
    num_heads = 4
    head_dim = 128
    dtype = torch.float16
    dtype_size = 2  # bytes for float16

    # Test with multiple blocks and layers (closer to real scenario)
    kv_count = 2  # K and V
    num_layers = 94  # Actual number of layers
    num_blocks = 13  # blocks 0-12, but valid blocks start from 1

    print("\n1. Starting peer agents...")
    try:
        agent1 = start_peer_agent(
            alias="test_agent_1",
            server_url="http://127.0.0.1:3000",
            device=None,
            ib_port=1,
            link_type="RoCE",
            qp_num=1,
        )
        print("   ✓ Agent 1 started")

        agent2 = start_peer_agent(
            alias="test_agent_2",
            server_url="http://127.0.0.1:3000",
            device=None,
            ib_port=1,
            link_type="RoCE",
            qp_num=1,
        )
        print("   ✓ Agent 2 started")
    except Exception as e:
        print(f"   ✗ Failed to start agents: {e}")
        return False

    print("\n2. Setting desired topology and waiting for connection...")
    try:
        agent1.set_desired_topology(target_peers=["test_agent_2"])
        agent2.set_desired_topology(target_peers=["test_agent_1"])
        agent1.wait_for_peers(["test_agent_2"])
        agent2.wait_for_peers(["test_agent_1"])
        print("   ✓ Connection initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize connection: {e}")
        return False

    print("\n3. Creating KV cache tensors...")
    # Agent 1 (source/remote): Create KV cache with known values
    kv_cache_1 = torch.randn(
        kv_count, num_layers, num_blocks, block_size, num_heads, head_dim, dtype=dtype
    )
    # Fill multiple blocks with distinct patterns for verification
    # This helps verify that we're reading from the correct locations
    for kv_idx in range(kv_count):
        for layer_idx in range(min(3, num_layers)):  # Test first 3 layers
            for block_idx in [1, 2, 3]:  # Test blocks 1, 2, 3
                if block_idx < num_blocks:
                    # Create a unique pattern for each (kv_idx, layer_idx, block_idx)
                    pattern_value = kv_idx * 1000 + layer_idx * 100 + block_idx
                    pattern = torch.full(
                        (block_size, num_heads, head_dim),
                        pattern_value,
                        dtype=torch.float32,
                    )
                    # Add some variation to make it more realistic
                    pattern = pattern + torch.randn_like(pattern) * 0.1
                    kv_cache_1[kv_idx, layer_idx, block_idx, :, :, :] = pattern.to(
                        dtype
                    )

    print(
        f"   Filled blocks with patterns: kv_idx=[0,1], layer_idx=[0,1,2], block_idx=[1,2,3]"
    )

    # Agent 2 (destination/local): Create empty KV cache
    kv_cache_2 = torch.zeros(
        kv_count, num_layers, num_blocks, block_size, num_heads, head_dim, dtype=dtype
    )

    print(f"   Source KV cache shape: {kv_cache_1.shape}")
    print(f"   Source block 1 mean: {kv_cache_1[0, 0, 1, :, :, :].mean().item():.6f}")
    print(f"   Destination KV cache shape: {kv_cache_2.shape}")
    print(
        f"   Destination block 1 mean: {kv_cache_2[0, 0, 1, :, :, :].mean().item():.6f}"
    )

    print("\n4. Registering memory regions...")
    try:
        # Register source KV cache on agent 1
        local_mr_1 = agent1.register_memory_region(
            "kv_cache",
            kv_cache_1.data_ptr() + kv_cache_1.storage_offset() * dtype_size,
            kv_cache_1.numel() * dtype_size,
        )
        print(f"   ✓ Agent 1 registered local MR: handler={local_mr_1}")

        # Register destination KV cache on agent 2
        local_mr_2 = agent2.register_memory_region(
            "kv_cache",
            kv_cache_2.data_ptr() + kv_cache_2.storage_offset() * dtype_size,
            kv_cache_2.numel() * dtype_size,
        )
        print(f"   ✓ Agent 2 registered local MR: handler={local_mr_2}")

        # Get remote MR info and register on agent 2
        remote_mr_info = agent2.get_mr_info("test_agent_1", "kv_cache")
        if remote_mr_info is None:
            print("   ✗ Failed to get remote MR info")
            return False

        remote_mr_2 = agent2.register_remote_memory_region(
            "test_agent_1",
            "kv_cache",
            remote_mr_info,
        )
        print(f"   ✓ Agent 2 registered remote MR: handler={remote_mr_2}")

    except Exception as e:
        print(f"   ✗ Failed to register memory regions: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n5. Calculating offsets...")
    # Calculate offset for block 1 (kv_idx=0, layer_idx=0, block_idx=1)
    block_stride = block_size * num_heads * head_dim * dtype_size
    block_idx = 1  # First valid block

    # Offset calculation (same as in cache.py)
    def block_stride_func(block_idx):
        return block_idx * block_size * num_heads * head_dim * dtype_size

    def local_layer_stride(layer_idx, block_idx):
        return block_stride_func(num_blocks) * layer_idx + block_stride_func(block_idx)

    def local_kv_stride(kv_idx, layer_idx, block_idx):
        return local_layer_stride(num_layers, 0) * kv_idx + local_layer_stride(
            layer_idx, block_idx
        )

    length = block_stride_func(1)

    print(f"   Block stride: {block_stride} bytes")
    print(f"   Length per block: {length} bytes")

    print("\n6. Getting endpoint...")
    try:
        endpoint = agent2.get_endpoint("test_agent_1")
        if endpoint is None:
            print("   ✗ Failed to get endpoint")
            return False
        print("   ✓ Endpoint obtained")
    except Exception as e:
        print(f"   ✗ Failed to get endpoint: {e}")
        return False

    print("\n7. Performing multiple RDMA reads...")
    # Test multiple blocks to ensure correctness
    test_cases = [
        (0, 0, 1),  # kv_idx=0, layer_idx=0, block_idx=1
        (0, 1, 1),  # kv_idx=0, layer_idx=1, block_idx=1
        (0, 0, 2),  # kv_idx=0, layer_idx=0, block_idx=2
        (1, 0, 1),  # kv_idx=1, layer_idx=0, block_idx=1 (V cache)
    ]

    print(f"   Testing {len(test_cases)} blocks...")
    slots = []
    for kv_idx, layer_idx, block_idx in test_cases:
        # Calculate offset for this block
        test_remote_off = local_kv_stride(kv_idx, layer_idx, block_idx)
        test_local_off = local_kv_stride(kv_idx, layer_idx, block_idx)

        # Perform RDMA read for this block
        try:
            rdma_op = [
                (local_mr_2, remote_mr_2, test_remote_off, test_local_off, length)
            ]
            slot = endpoint.read(rdma_op, None)
            if slot is None:
                print(
                    f"     ✗ endpoint.read returned None for kv={kv_idx}, layer={layer_idx}, block={block_idx}"
                )
                return False
            slots.append((slot, kv_idx, layer_idx, block_idx))
        except Exception as e:
            print(
                f"     ✗ RDMA read failed for kv={kv_idx}, layer={layer_idx}, block={block_idx}: {e}"
            )
            return False

    # Wait for all reads to complete
    print("   Waiting for all RDMA reads to complete...")
    for slot, kv_idx, layer_idx, block_idx in slots:
        slot.wait()
    print("   ✓ All RDMA reads completed")

    print("\n8. Verifying read data...")
    all_passed = True
    for kv_idx, layer_idx, block_idx in test_cases:
        print(
            f"\n   Testing: kv_idx={kv_idx}, layer_idx={layer_idx}, block_idx={block_idx}"
        )

        # Get the source and destination block data
        source_block = kv_cache_1[kv_idx, layer_idx, block_idx, :, :, :]
        dest_block = kv_cache_2[kv_idx, layer_idx, block_idx, :, :, :]

        print(
            f"     Source mean: {source_block.mean().item():.6f}, std: {source_block.std().item():.6f}"
        )
        print(
            f"     Dest mean: {dest_block.mean().item():.6f}, std: {dest_block.std().item():.6f}"
        )

        # Compare (handle NaN/Inf cases)
        diff = dest_block - source_block
        valid_mask = torch.isfinite(diff)

        if not torch.all(valid_mask):
            num_invalid = (~valid_mask).sum().item()
            print(
                f"     ⚠ Warning: {num_invalid} invalid values (NaN/Inf) in difference"
            )

        # Check if data matches (only compare valid values)
        if torch.all(valid_mask):
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()

            if torch.allclose(dest_block, source_block, atol=1e-2, rtol=1e-2):
                print(
                    f"     ✓ Block matches! Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}"
                )
            else:
                print(
                    f"     ✗ Block does NOT match! Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}"
                )

                # Show first few elements for debugging
                print(f"     First 10 elements:")
                print(f"       Source: {source_block.flatten()[:10].cpu().numpy()}")
                print(f"       Dest:   {dest_block.flatten()[:10].cpu().numpy()}")
                print(f"       Diff:   {diff.flatten()[:10].cpu().numpy()}")

                all_passed = False
        else:
            # Compare only valid values
            valid_diff = diff[valid_mask]
            valid_source = source_block[valid_mask]
            valid_dest = dest_block[valid_mask]

            max_diff = torch.max(torch.abs(valid_diff)).item()
            mean_diff = torch.mean(torch.abs(valid_diff)).item()

            if torch.allclose(valid_dest, valid_source, atol=1e-2, rtol=1e-2):
                print(
                    f"     ✓ Block matches (valid values)! Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}"
                )
            else:
                print(
                    f"     ✗ Block does NOT match! Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}"
                )
                all_passed = False

    if all_passed:
        print("\n   ✓ All test blocks passed! RDMA read is working correctly.")
        return True
    else:
        print("\n   ✗ Some test blocks failed!")
        return False


if __name__ == "__main__":
    print("Running Peer Agent RDMA Read Test\n")

    try:
        success = test_peer_agent_rdma_read()
        if success:
            print("\n" + "=" * 80)
            print("✓ Test passed!")
            print("=" * 80)
            exit(0)
        else:
            print("\n" + "=" * 80)
            print("✗ Test failed!")
            print("=" * 80)
            exit(1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\nTest failed with exception: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
