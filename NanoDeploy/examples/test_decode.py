import argparse
import json
import time

import ray
import torch
from test_utils import BLOCK_SIZE, NUM_KVS, NUM_LAYERS, TestWorker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-ranks", type=int, default=1, help="Number of ranks")
    parser.add_argument("--base-port", type=int, default=31000, help="Base port")
    args = parser.parse_args()

    # Cleanup ports before starting
    from test_utils import cleanup_ports

    base_port = args.base_port
    cleanup_ports(base_port, args.num_ranks)

    ray.init(ignore_reinit_error=True)

    num_ranks = args.num_ranks
    host = "127.0.0.1"

    # Start decode workers
    workers = [TestWorker.remote(i, host, base_port) for i in range(num_ranks)]

    # Load prefill addresses
    try:
        with open("prefill_addrs.json", "r") as f:
            prefill_addrs = json.load(f)
    except FileNotFoundError:
        print("Error: prefill_addrs.json not found. Run test_prefill.py first.")
        return

    print("\n" + "=" * 50)
    print("STARTING RDMA TESTS (8x8)")
    print("=" * 50)

    # Test Case 1: Simple 1-to-1 Mapping
    print("\n[Test 1] Simple 1-to-1 Mapping...")
    ray.get([w.clear_cache.remote() for w in workers])

    # Each decode rank i reads block 10 from prefill rank i
    # Offset calculation: block index * BLOCK_SIZE
    b_idx = 10
    length = BLOCK_SIZE
    off = b_idx * BLOCK_SIZE

    tasks = []
    for i in range(num_ranks):
        assigns = [(off, off, length)]
        tasks.append(workers[i].rdma_read.remote(prefill_addrs[i], assigns))

    ray.get(tasks)

    results = ray.get(
        [
            workers[i].verify_data.remote([(off, off, length)], i)
            for i in range(num_ranks)
        ]
    )
    if all(results):
        print("SUCCESS: 1-to-1 Mapping")
    else:
        print("FAILED: 1-to-1 Mapping")

    # Test Case 2: One rank reads multiple blocks from multiple prefill ranks
    print("\n[Test 2] Scattered Read (Rank 0 reads from all Prefill ranks)...")
    ray.get(workers[0].clear_cache.remote())

    # Rank 0 reads block 20 from each prefill rank 0-7
    # We'll map them to local blocks 0-7 to avoid overlap
    scattered_assigns = []
    for i in range(num_ranks):
        remote_off = 20 * BLOCK_SIZE
        local_off = i * BLOCK_SIZE
        # Execute individual reads per remote host
        ray.get(
            workers[0].rdma_read.remote(
                prefill_addrs[i], [(local_off, remote_off, length)]
            )
        )
        scattered_assigns.append(
            (local_off, remote_off, i)
        )  # (l_off, r_off, expected_rank)

    # Verify each one
    verifications = [
        workers[0].verify_data.remote([(l_off, r_off, length)], exp_rank)
        for l_off, r_off, exp_rank in scattered_assigns
    ]
    if all(ray.get(verifications)):
        print("SUCCESS: Scattered Read")
    else:
        print("FAILED: Scattered Read")

    # Test Case 3: Many-to-One (All Decode ranks read from Prefill Rank 0)
    print("\n[Test 3] Contention Test (All Decode ranks read from Prefill Rank 0)...")
    ray.get([w.clear_cache.remote() for w in workers])

    tasks = []
    off = 30 * BLOCK_SIZE
    for i in range(num_ranks):
        tasks.append(
            workers[i].rdma_read.remote(prefill_addrs[0], [(off, off, length)])
        )
    ray.get(tasks)

    results = ray.get(
        [
            workers[i].verify_data.remote([(off, off, length)], 0)
            for i in range(num_ranks)
        ]
    )
    if all(results):
        print("SUCCESS: Contention Test")
    else:
        print("FAILED: Contention Test")

    # Test Case 4: Batch Read (One future, one rank reads multiple blocks from one rank)
    print("\n[Test 4] Batch Read (Rank 0 reads 3 blocks from Prefill Rank 0)...")
    ray.get(workers[0].clear_cache.remote())

    # Read remote blocks 0, 1, 2 to local blocks 0, 1, 2
    batch_assigns = []
    expected_results = []
    for b in range(3):
        l_off = b * BLOCK_SIZE
        r_off = b * BLOCK_SIZE
        batch_assigns.append((l_off, r_off, length))

    # Single read call with multiple assignments -> One future
    ray.get(workers[0].rdma_read.remote(prefill_addrs[0], batch_assigns))

    # Verify
    if ray.get(workers[0].verify_data.remote(batch_assigns, 0)):
        print("SUCCESS: Batch Read")
    else:
        print("FAILED: Batch Read")

    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()
