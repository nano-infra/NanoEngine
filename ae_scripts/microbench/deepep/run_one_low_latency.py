#!/usr/bin/env python3
"""Run DeepEP's stock low-latency test for one configurable token count.

This adapter uses the DeepEP benchmark sources vendored with the artifact and
supplies a configurable launcher and buffer setup for one token count.

DeepEP's tests/utils.py interprets WORLD_SIZE as the number of nodes and RANK
as the node rank.  Each node then spawns EP_TEST_NUM_PROCESSES local ranks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


DEEPEP_TESTS = Path(__file__).resolve().parent / "third_party" / "deepep"

if not (DEEPEP_TESTS / "test_low_latency.py").is_file():
    raise FileNotFoundError(
        f"DeepEP low-latency test not found: {DEEPEP_TESTS / 'test_low_latency.py'}"
    )

# Make the vendored test module and its sibling ``utils.py`` importable.  The
# compiled ``deep_ep`` package itself must be installed in the environment.
sys.path.insert(0, str(DEEPEP_TESTS))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import deep_ep  # noqa: E402
from test_low_latency import test_main  # noqa: E402
from utils import init_dist  # noqa: E402


def env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def test_loop(local_rank: int, num_local_ranks: int) -> None:
    node_rank = int(os.environ.get("RANK", "0"))
    num_nodes = env_int("WORLD_SIZE", 1)
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "8361")
    if local_rank == 0:
        first_rank = node_rank * num_local_ranks
        print(
            f"[node {node_rank}] initializing NCCL ranks "
            f"{first_rank}..{first_rank + num_local_ranks - 1} of "
            f"{num_nodes * num_local_ranks} via "
            f"tcp://{master_addr}:{master_port}",
            flush=True,
        )

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    # Initialize the default NCCL communicator while every rank is aligned.
    # DeepEP's profiler later uses the default group for an all-reduce; doing
    # the first collective here avoids lazy communicator setup in the middle
    # of profiling.
    dist.all_reduce(torch.ones(1, dtype=torch.float, device="cuda"))
    dist.barrier(group=group)
    if local_rank == 0:
        print(f"[node {node_rank}] NCCL initialization completed", flush=True)

    num_tokens = env_int("EP_TEST_NUM_TOKENS", 128)
    hidden = env_int("EP_TEST_HIDDEN", 7168)
    num_topk = env_int("EP_TEST_NUM_TOPK", 8)
    # DeepSeek-V3 has 256 routed experts.  Do not inherit DeepEP's generic
    # test_low_latency.py default of 288, which is not a model configuration.
    num_experts = env_int("EP_TEST_NUM_EXPERTS", 256)
    seed = int(os.environ.get("EP_TEST_SEED", "1"))

    if num_experts % num_ranks != 0:
        raise ValueError(
            f"EP_TEST_NUM_EXPERTS={num_experts} must be divisible by "
            f"the total rank count ({num_ranks})"
        )

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, num_ranks, num_experts
    )
    if local_rank == 0:
        print(
            f"[sweep config] tokens={num_tokens}, hidden={hidden}, topk={num_topk}, "
            f"experts={num_experts}, ranks={num_ranks}, seed={seed}",
            flush=True,
        )
        print(f"Allocating buffer size: {num_rdma_bytes / 1e6} MB ...", flush=True)

    buffer = deep_ep.Buffer(
        group,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_experts // num_ranks,
        explicitly_destroy=True,
    )
    try:
        test_main(
            num_tokens,
            hidden,
            num_experts,
            num_topk,
            rank,
            num_ranks,
            group,
            buffer,
            seed=seed,
        )
    finally:
        buffer.destroy()

    # Explicit teardown makes repeated one-token invocations reliable.
    dist.barrier(group=group)
    dist.destroy_process_group()


def main() -> None:
    num_processes = env_int("EP_TEST_NUM_PROCESSES", 8)
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes,), nprocs=num_processes
    )


if __name__ == "__main__":
    main()
