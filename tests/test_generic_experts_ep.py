"""Generic experts expert-parallel (ep_size > 1) correctness.

Runs a 2-rank gloo group and asserts the reference all-gather EP path matches a
single-rank full-expert computation. CPU-only; skipped if spawn/gloo is
unavailable.
"""

import os

import pytest
import torch
import torch.multiprocessing as mp


def _worker(rank, world, port, q):
    try:
        import torch.distributed as dist

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=rank, world_size=world)
        from dlengine.runtime.layers.backends.generic.experts import (
            GenericDistributedRoutedExperts,
        )

        H, I, E, K = 8, 8, 4, 2
        ep_group = dist.group.WORLD
        exp = GenericDistributedRoutedExperts(
            hidden_size=H,
            intermediate_size=I,
            num_experts=E,
            top_k=K,
            ep_size=world,
            tp_size=1,
            ep_group=ep_group,
        )
        torch.manual_seed(1234)
        full_gate_up = torch.randn(E, I * 2, H, dtype=torch.bfloat16)
        full_down = torch.randn(E, H, I, dtype=torch.bfloat16)
        local = E // world
        exp.gate_up_proj.data.copy_(full_gate_up[rank * local : (rank + 1) * local])
        exp.down_proj.data.copy_(full_down[rank * local : (rank + 1) * local])

        torch.manual_seed(100 + rank)
        x = torch.randn(3 + rank, H, dtype=torch.bfloat16)
        ids = torch.randint(0, E, (x.shape[0], K), dtype=torch.long)
        w = torch.rand(x.shape[0], K)
        out = exp(x, ids, w, is_prefill=True)

        ref = GenericDistributedRoutedExperts(
            hidden_size=H,
            intermediate_size=I,
            num_experts=E,
            top_k=K,
            ep_size=1,
            tp_size=1,
        )
        ref.gate_up_proj.data.copy_(full_gate_up)
        ref.down_proj.data.copy_(full_down)
        ref_out = ref(x, ids, w, is_prefill=True)

        ok = torch.allclose(out.float(), ref_out.float(), atol=1e-2, rtol=1e-2)
        q.put((rank, ok))
        dist.destroy_process_group()
    except Exception as exc:  # pragma: no cover - surfaced via queue
        import traceback

        q.put((rank, f"ERR: {traceback.format_exc()}"))


def test_generic_experts_ep_matches_single_rank():
    world = 2
    try:
        ctx = mp.get_context("spawn")
    except Exception:  # pragma: no cover
        pytest.skip("spawn start method unavailable")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(r, world, 29790, q)) for r in range(world)
    ]
    for p in procs:
        p.start()
    results = [q.get(timeout=90) for _ in range(world)]
    for p in procs:
        p.join(timeout=10)

    for rank, ok in results:
        assert ok is True, f"rank {rank}: {ok}"
