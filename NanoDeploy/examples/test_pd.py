#!/usr/bin/env python3
"""
Unified PD Disaggregation RDMA Test
Runs both prefill and decode workers in a single script on one machine.
Automatically finds free ports to avoid conflicts.
"""

import argparse
import socket
import time

import ray
from test_utils import BLOCK_SIZE, cleanup_ports, TestWorker


def find_free_port(start=30000):
    """Find a free port starting from the given number."""
    for port in range(start, start + 1000):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", port))
            s.close()
            return port
        except OSError:
            continue
    raise RuntimeError("Could not find free port")


def main():
    parser = argparse.ArgumentParser(description="Unified PD RDMA Test")
    parser.add_argument(
        "--num-ranks", type=int, default=1, help="Number of ranks per side"
    )
    args = parser.parse_args()

    num_ranks = args.num_ranks
    host = "127.0.0.1"

    # Find free port ranges for prefill and decode
    prefill_base = find_free_port(30000)
    decode_base = find_free_port(prefill_base + num_ranks + 10)

    print(f"Using prefill ports: {prefill_base}-{prefill_base + num_ranks - 1}")
    print(f"Using decode ports: {decode_base}-{decode_base + num_ranks - 1}")

    # Cleanup just in case
    cleanup_ports(prefill_base, num_ranks)
    cleanup_ports(decode_base, num_ranks)

    # Shutdown any existing Ray connection and start a fresh LOCAL cluster
    # This ensures all actors run on THIS machine (127.0.0.1 works correctly)
    ray.shutdown()
    ray.init()  # Start fresh local cluster, don't connect to existing one

    # Start prefill workers
    print("\n[Starting Prefill Workers]")
    prefill_workers = [
        TestWorker.remote(i, host, prefill_base) for i in range(num_ranks)
    ]
    prefill_addrs = ray.get([w.get_addr.remote() for w in prefill_workers])
    print(f"Prefill addresses: {prefill_addrs}")

    # Start decode workers
    print("\n[Starting Decode Workers]")
    decode_workers = [TestWorker.remote(i, host, decode_base) for i in range(num_ranks)]
    decode_addrs = ray.get([w.get_addr.remote() for w in decode_workers])
    print(f"Decode addresses: {decode_addrs}")

    print("\n" + "=" * 50)
    print(f"RDMA TESTS ({num_ranks}x{num_ranks})")
    print("=" * 50)

    length = BLOCK_SIZE

    # Test 1: 1-to-1 Mapping
    print("\n[Test 1] 1-to-1 Mapping...")
    ray.get([w.clear_cache.remote() for w in decode_workers])
    off = 10 * BLOCK_SIZE
    tasks = [
        decode_workers[i].rdma_read.remote(prefill_addrs[i], [(off, off, length)])
        for i in range(num_ranks)
    ]
    ray.get(tasks)
    results = ray.get(
        [
            decode_workers[i].verify_data.remote([(off, off, length)], i)
            for i in range(num_ranks)
        ]
    )
    print("SUCCESS" if all(results) else "FAILED")

    # Test 2: Scattered Read (Rank 0 reads from all prefill)
    print("\n[Test 2] Scattered Read...")
    ray.get(decode_workers[0].clear_cache.remote())
    for i in range(num_ranks):
        remote_off = 20 * BLOCK_SIZE
        local_off = i * BLOCK_SIZE
        ray.get(
            decode_workers[0].rdma_read.remote(
                prefill_addrs[i], [(local_off, remote_off, length)]
            )
        )
    results = [
        ray.get(
            decode_workers[0].verify_data.remote(
                [(i * BLOCK_SIZE, 20 * BLOCK_SIZE, length)], i
            )
        )
        for i in range(num_ranks)
    ]
    print("SUCCESS" if all(results) else "FAILED")

    # Test 3: Contention (All decode read from prefill 0)
    print("\n[Test 3] Contention...")
    ray.get([w.clear_cache.remote() for w in decode_workers])
    off = 30 * BLOCK_SIZE
    tasks = [
        decode_workers[i].rdma_read.remote(prefill_addrs[0], [(off, off, length)])
        for i in range(num_ranks)
    ]
    ray.get(tasks)
    results = ray.get(
        [
            decode_workers[i].verify_data.remote([(off, off, length)], 0)
            for i in range(num_ranks)
        ]
    )
    print("SUCCESS" if all(results) else "FAILED")

    # Test 4: Batch Read
    print("\n[Test 4] Batch Read...")
    ray.get(decode_workers[0].clear_cache.remote())
    batch = [(b * BLOCK_SIZE, b * BLOCK_SIZE, length) for b in range(3)]
    ray.get(decode_workers[0].rdma_read.remote(prefill_addrs[0], batch))
    result = ray.get(decode_workers[0].verify_data.remote(batch, 0))
    print("SUCCESS" if result else "FAILED")

    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()
