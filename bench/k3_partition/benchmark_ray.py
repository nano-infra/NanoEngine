#!/usr/bin/env python3
"""Ray-launched GB200 components, collectives and real-checkpoint layer measurements."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_gpus=1, num_cpus=2)
class Worker:
    def __init__(self, rank, root):
        import os
        import sys
        import torch

        sys.path.insert(0, root)
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        torch.set_num_threads(2)
        torch.cuda.set_device(0)
        self.rank = rank
        self.world = 16

    def inventory(self):
        import os
        import socket
        import subprocess
        import torch

        p = torch.cuda.get_device_properties(0)
        return dict(rank=self.rank, host=socket.gethostname(),
                    visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    gpu=p.name, hbm_bytes=p.total_memory,
                    torch=torch.__version__, cuda=torch.version.cuda,
                    nccl=torch.cuda.nccl.version(),
                    topology=subprocess.run(["nvidia-smi", "topo", "-m"],
                        capture_output=True, text=True).stdout)

    def rendezvous(self):
        import socket
        from ray.util import get_node_ip_address

        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        return f"tcp://{get_node_ip_address()}:{port}"

    def initialize(self, address, world):
        import datetime
        import torch.distributed as dist
        import torch

        self.world = world
        dist.init_process_group("nccl", init_method=address, rank=self.rank,
                                world_size=world,
                                timeout=datetime.timedelta(minutes=4), device_id=torch.device("cuda", 0))
        self.groups = {}
        # All independent groups run simultaneously, matching a full mesh.
        for size in (1, 2, 4, 8, 16):
            if size > world:
                continue
            for start in range(0, world, size):
                ranks = list(range(start, start + size))
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self.groups[size] = group
        dist.barrier()
        return self.rank

    @staticmethod
    def time(op, warmup=5, repeats=30):
        import torch

        for _ in range(warmup):
            op()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(repeats):
            op()
        b.record()
        b.synchronize()
        return a.elapsed_time(b) / repeats

    def roofline(self):
        import torch

        rows = []
        # The working set exceeds L2; report read + write bytes for copy.
        a = torch.empty(256 * 2**20, device="cuda", dtype=torch.uint8).fill_(1)
        b = torch.empty_like(a)
        ms = self.time(lambda: b.copy_(a), repeats=100)
        rows.append(dict(rank=self.rank, op="hbm_copy", m=0, n=0, k=0,
                         tp=1, ms=ms, gflops=0,
                         gbps=2 * a.numel() / ms / 1e6))
        del a, b
        for tp in (1, 2, 4, 8, 16):
            for m in (1, 8, 128, 1024, 8192):
                n, k = 12288 // tp, 7168
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
                out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                ms = self.time(lambda: torch.mm(x, w.T, out=out))
                rows.append(dict(rank=self.rank, op="bf16_kda_projection", m=m,
                                 n=n, k=k, tp=tp, ms=ms,
                                 gflops=2 * m * n * k / ms / 1e6, gbps=0))
                del x, w, out
        torch.cuda.empty_cache()
        return rows

    def collectives(self, tokens, warmup, repeats, graph_mode=False):
        import torch
        import torch.distributed as dist

        rows = []
        for size, group in self.groups.items():
            if size == 1:
                continue
            for requested in tokens:
                padded = (requested + size - 1) // size * size
                full = torch.zeros(padded, 7168, device="cuda", dtype=torch.bfloat16)
                shard = torch.zeros(padded // size, 7168, device="cuda", dtype=torch.bfloat16)
                gathered, reduced, exchanged = torch.empty_like(full), torch.empty_like(shard), torch.empty_like(full)
                ops = {
                    "all_reduce": lambda: dist.all_reduce(full, group=group),
                    "reduce_scatter": lambda: dist.reduce_scatter_tensor(reduced, full, group=group),
                    "all_gather": lambda: dist.all_gather_into_tensor(gathered, shard, group=group),
                    "all_to_all": lambda: dist.all_to_all_single(exchanged, full, group=group),
                    "rs_ag_pair": lambda: (
                        dist.reduce_scatter_tensor(reduced, full, group=group),
                        dist.all_gather_into_tensor(gathered, reduced, group=group)),
                }
                # Verify rank ownership on nonzero data before timing zeros.
                local_rank = dist.get_rank(group)
                full.fill_(local_rank + 1)
                ops["reduce_scatter"]()
                torch.cuda.synchronize()
                assert torch.all(reduced == size * (size + 1) // 2).item()
                shard.fill_(local_rank + 1)
                ops["all_gather"]()
                torch.cuda.synchronize()
                expected = torch.arange(1, size + 1, device="cuda", dtype=full.dtype).repeat_interleave(padded // size)
                assert torch.all(gathered[:, 0] == expected).item()
                full.zero_()
                for name, op in ops.items():
                    divisor = 1
                    if graph_mode:
                        dist.barrier()
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(5):
                                op()
                        stream.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        dist.barrier()
                        with torch.cuda.graph(graph, stream=stream):
                            for _ in range(8):
                                op()
                        op = graph.replay
                        divisor = 8
                    trials = []
                    for _ in range(3):
                        dist.barrier()
                        trials.append(self.time(op, warmup, repeats) / divisor)
                    rows.append(dict(rank=self.rank, group_start=self.rank // size * size,
                        size=size, requested_tokens=requested, padded_tokens=padded,
                        op=name, logical_bytes=full.numel() * 2,
                        median_ms=sorted(trials)[1], min_ms=min(trials), max_ms=max(trials)))
                del full, shard, gathered, reduced, exchanged
        torch.cuda.empty_cache()
        return rows

    def components(self, kind):
        import faulthandler
        faulthandler.dump_traceback_later(120, repeat=True)
        try:
            from bench.k3_partition.component_kernels import measure_kda, measure_cp, measure_moe, measure_transition, measure_validation
            from bench.k3_partition.extended_kernels import measure_checkpoint, measure_mla, measure_real_moe, measure_equal_kda, measure_equal_mla, measure_mla_tp1, measure_cold, measure_cold_trace
            from bench.k3_partition.paged_cp import measure_paged_cp
            from bench.k3_partition.expert_tp import measure_expert_tp
            return {"kda": measure_kda, "cp": measure_cp, "moe": measure_moe, "transition": measure_transition, "validation": measure_validation, "checkpoint": measure_checkpoint, "mla": measure_mla, "paged_cp": measure_paged_cp, "real_moe": measure_real_moe, "equal_kda": measure_equal_kda, "equal_mla": measure_equal_mla, "mla_tp1": measure_mla_tp1, "cold": measure_cold, "cold_trace": measure_cold_trace, "expert_tp": measure_expert_tp}[kind](self)
        finally:
            faulthandler.cancel_dump_traceback_later()

    def close(self):
        import torch.distributed as dist
        dist.destroy_process_group()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results/gb200")
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 8, 128, 1024, 8192, 16384])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--stages", nargs="+", choices=["baseline", "graphs", "kda", "cp", "moe", "transition", "validation", "checkpoint", "mla", "paged_cp", "real_moe", "equal_kda", "equal_mla", "mla_tp1", "cold", "cold_trace", "expert_tp"], default=["baseline"])
    parser.add_argument("--world-size", type=int, choices=[1, 2, 4, 8, 16], default=16)
    parser.add_argument("--gpu-offset", type=int, default=0, help="Skip this many GPUs in sorted-node allocation order")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ray.init(address="auto")
    nodes = sorted((n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0)), key=lambda n: n["NodeManagerAddress"])
    if sum(int(n["Resources"]["GPU"]) for n in nodes) < args.world_size + args.gpu_offset:
        raise RuntimeError("Insufficient Ray GPU resources")
    if args.world_size != 16 and any(s in args.stages for s in ("cp", "moe", "transition", "validation", "paged_cp", "real_moe", "cold", "cold_trace", "expert_tp")):
        parser.error("Legacy component stages require world size 16")
    root = str(Path(__file__).resolve().parents[2])
    workers = []
    try:
        skip = args.gpu_offset
        for node in nodes:
            available = int(node["Resources"]["GPU"])
            omitted = min(skip, available)
            skip -= omitted
            for _ in range(min(available - omitted, args.world_size - len(workers))):
                workers.append(Worker.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)).remote(len(workers), root))
        inventory = ray.get([w.inventory.remote() for w in workers], timeout=180)
        import datetime
        metadata = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        nodes=[dict(ip=n["NodeManagerAddress"], id=n["NodeID"]) for n in nodes],
                        world_size=args.world_size, workers=inventory, warmup=args.warmup, repeats=args.repeats,
                        trials=3, tokens=args.tokens)
        import hashlib
        source_paths = list(Path(root, "bench/k3_partition").glob("*.py")) + [
            Path(root, "dlengine/runtime/layers/backends/mla/trtllm.py"),
            Path(root, "dlengine/runtime/layers/backends/experts/mega_moe.py"),
            Path(root, "dlengine/runtime/models/deepseek_v2/deepseek_v2.py")]
        metadata['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
        (args.output_dir / ("environment_" + "_".join(args.stages) + ".json")).write_text(json.dumps(metadata, indent=2) + "\n")
        if "baseline" in args.stages:
            print(f"Measuring HBM and GEMMs on {args.world_size} GPUs", flush=True)
            rows = ray.get([w.roofline.remote() for w in workers], timeout=600)
            write_csv(args.output_dir / "roofline.csv", sum(rows, []))
        address = ray.get(workers[0].rendezvous.remote())
        ray.get([w.initialize.remote(address, len(workers)) for w in workers], timeout=300)
        if "baseline" in args.stages:
            print(f"Measuring collectives on {args.world_size} GPUs", flush=True)
            rows = ray.get([w.collectives.remote(args.tokens, args.warmup, args.repeats) for w in workers], timeout=1800)
            write_csv(args.output_dir / "collectives.csv", sum(rows, []))
        for kind in args.stages:
            if kind == "baseline":
                continue
            print(f"Measuring {kind} on {args.world_size} GPUs", flush=True)
            if kind == "graphs":
                rows = ray.get([w.collectives.remote(args.tokens, args.warmup, args.repeats, True) for w in workers], timeout=1800)
            else:
                rows = ray.get([w.components.remote(kind) for w in workers], timeout=2400)
            write_csv(args.output_dir / f"{kind}.csv", sum(rows, []))
            print(f"Saved {kind}.csv", flush=True)
        ray.get([w.close.remote() for w in workers], timeout=60)
        print(f"Results written to {args.output_dir}", flush=True)
    finally:
        for worker in workers:
            ray.kill(worker)
        ray.shutdown()


if __name__ == "__main__":
    main()
