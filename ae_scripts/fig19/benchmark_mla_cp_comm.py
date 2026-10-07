import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from dlslime.buffer.intra.all_to_all_intra_ll_buffer import AllToAllIntraLLBuffer


MASTER_RANK = 0


@dataclass(frozen=True)
class PayloadSpec:
    name: str
    feature_dim: int
    traffic_pattern: str
    is_transpose: bool
    max_dispatch_per_msg: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate single-request DeepSeek-V3 MLA SP communication cost across CP sizes"
    )
    parser.add_argument(
        "--cp-sizes",
        type=str,
        default="1,2,4,8",
        help="Comma-separated CP sizes to benchmark. Use torchrun with at least max(cp_sizes) ranks.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--num-heads", type=int, default=128, help="DeepSeek-V3 MLA query heads")
    parser.add_argument("--head-dim", type=int, default=576, help="MLA Q/K head dim")
    parser.add_argument("--v-head-dim", type=int, default=512, help="MLA output V dim")
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1,
        help="How many logical requests/messages are carried in the target MLA communication.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=2,
        help="SP buffer/layout max_num_seqs. Use 2 to match NanoDeploy-new single-request MLA benchmark.",
    )
    parser.add_argument("--mode", type=str, default="graph", choices=["eager", "graph"])
    parser.add_argument("--preamble", type=str, default="all_reduce", choices=["none", "all_reduce", "all_gather"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-iters", type=int, default=10)
    parser.add_argument("--graph-inner-iters", type=int, default=20)
    parser.add_argument("--trace-dir", type=str, default="profiler_traces/mla_single_request_cp_comm")
    parser.add_argument(
        "--summary-path",
        type=str,
        default="profiler_traces/mla_single_request_cp_comm/summary.json",
    )
    return parser.parse_args()


def parse_cp_sizes(cp_sizes_text):
    values = []
    for part in cp_sizes_text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"CP size must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("No CP sizes provided")
    return sorted(dict.fromkeys(values))


def get_dtype(dtype_name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def percentile(values, q):
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, q / 100.0).item())


def summarize_values(values):
    if not values:
        return {
            "count": 0,
            "mean_us": 0.0,
            "p50_us": 0.0,
            "p90_us": 0.0,
            "p99_us": 0.0,
            "min_us": 0.0,
            "max_us": 0.0,
        }
    return {
        "count": len(values),
        "mean_us": float(sum(values) / len(values)),
        "p50_us": percentile(values, 50),
        "p90_us": percentile(values, 90),
        "p99_us": percentile(values, 99),
        "min_us": float(min(values)),
        "max_us": float(max(values)),
    }


def init_dist():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def make_group(cp_size):
    return dist.new_group(ranks=list(range(cp_size)), backend="nccl")


def make_row(fill_value, feature_dim, dtype, device):
    return torch.full((feature_dim,), fill_value=fill_value, dtype=dtype, device=device)


def shared_msg_size(num_heads, head_dim):
    return (head_dim + 1) * num_heads


def payload_specs(num_heads, head_dim, v_head_dim, cp_size):
    return [
        PayloadSpec(
            name="Q",
            feature_dim=num_heads * head_dim,
            traffic_pattern="one_to_many",
            is_transpose=False,
            max_dispatch_per_msg=cp_size,
        ),
        PayloadSpec(
            name="Res",
            feature_dim=num_heads * v_head_dim,
            traffic_pattern="many_to_one",
            is_transpose=True,
            max_dispatch_per_msg=1,
        ),
        PayloadSpec(
            name="Lse",
            feature_dim=num_heads,
            traffic_pattern="many_to_one",
            is_transpose=True,
            max_dispatch_per_msg=1,
        ),
    ]


def q_payload_value():
    return 1.0


def q_seq_value(seq_idx):
    return 1.0 + float(seq_idx)


def res_seq_value(cp_rank, seq_idx):
    return 128.0 + float(cp_rank * 16 + seq_idx)


def lse_seq_value(cp_rank, seq_idx):
    return float(cp_rank * 16 + seq_idx)


def build_q_setup(buffer, cp_rank, cp_size, max_num_seqs, num_requests, dtype, device, feature_dim):
    x = torch.zeros((num_requests, feature_dim), dtype=dtype, device=device)
    mask = torch.zeros((cp_size, max_num_seqs), dtype=torch.int32, device=device)

    buffer.local_buffer.zero_()
    if cp_rank == MASTER_RANK:
        for seq_idx in range(num_requests):
            x[seq_idx].copy_(make_row(q_seq_value(seq_idx), feature_dim, dtype, device))
            for dst in range(cp_size):
                mask[dst, seq_idx] = 1

    expected = torch.zeros((cp_size, max_num_seqs, feature_dim), dtype=dtype, device=device)
    for seq_idx in range(num_requests):
        expected[MASTER_RANK, seq_idx].copy_(make_row(q_seq_value(seq_idx), feature_dim, dtype, device))
    return x, mask, expected


def build_reduce_setup(buffer, cp_rank, cp_size, max_num_seqs, num_requests, dtype, device, feature_dim, value_fn):
    x = torch.zeros((cp_size * max_num_seqs, feature_dim), dtype=dtype, device=device)
    mask = torch.zeros((cp_size, max_num_seqs), dtype=torch.int32, device=device)
    buffer.local_buffer.zero_()
    for seq_idx in range(num_requests):
        row = make_row(value_fn(cp_rank, seq_idx), feature_dim, dtype, device)
        x[cp_rank * max_num_seqs + seq_idx].copy_(row)
        mask[cp_rank, seq_idx] = 1
        if cp_rank != MASTER_RANK:
            x[MASTER_RANK * max_num_seqs + seq_idx].copy_(row)
            mask[MASTER_RANK, seq_idx] = 1

    expected = torch.zeros((cp_size, max_num_seqs, feature_dim), dtype=dtype, device=device)
    for seq_idx in range(num_requests):
        expected[cp_rank, seq_idx].copy_(make_row(value_fn(cp_rank, seq_idx), feature_dim, dtype, device))
    if cp_rank == MASTER_RANK:
        for src in range(cp_size):
            for seq_idx in range(num_requests):
                expected[src, seq_idx].copy_(make_row(value_fn(src, seq_idx), feature_dim, dtype, device))
    return x, mask, expected


class BaseRunner:
    def __init__(self, group, preamble, device, inner_iters):
        self.group = group
        self.preamble = preamble
        self.device = device
        self.inner_iters = inner_iters
        self.sync_tensor = torch.tensor([1.0], dtype=torch.float32, device=device)
        self.gather_out = torch.empty(
            dist.get_world_size(group=group),
            dtype=self.sync_tensor.dtype,
            device=device,
        )

    def preamble_op(self):
        if self.preamble == "none":
            return
        if self.preamble == "all_reduce":
            dist.all_reduce(self.sync_tensor, op=dist.ReduceOp.SUM, group=self.group)
            self.sync_tensor.fill_(1.0)
            return
        if self.preamble == "all_gather":
            dist.all_gather_into_tensor(self.gather_out, self.sync_tensor, group=self.group)
            return
        raise ValueError(f"Unsupported preamble: {self.preamble}")


class LLRunner(BaseRunner):
    def __init__(self, group, preamble, device, inner_iters, buffer, x, mask, is_transpose):
        super().__init__(group=group, preamble=preamble, device=device, inner_iters=inner_iters)
        self.buffer = buffer
        self.x = x
        self.mask = mask
        self.is_transpose = is_transpose
        self.output = None

    def run_once(self):
        for _ in range(self.inner_iters):
            self.preamble_op()
            self.output = self.buffer.all_to_all_ll(self.x, is_transpose=self.is_transpose, mask=self.mask)


def capture_graph(runner, warmup, group, device):
    for _ in range(warmup):
        runner.run_once()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.run_once()
    return graph


def walltime_benchmark(runner, graph, mode, iters, device, group):
    dist.barrier(group=group)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        if mode == "graph":
            graph.replay()
        else:
            runner.run_once()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    end = time.perf_counter()
    total_ops = max(iters * runner.inner_iters, 1)
    return (end - start) * 1_000_000.0 / total_ops


def matches_all_to_all_kernel(low_name):
    return (
        "all_to_all_intra_ll_kernel" in low_name
        or "all_to_all_intra_ll" in low_name
        or ("alltoall" in low_name and ("intra" in low_name or "slime" in low_name))
    )


def kernel_stream_id(evt):
    args = evt.get("args", {})
    stream = args.get("stream", evt.get("tid"))
    if stream is None:
        return None
    try:
        return int(stream)
    except (TypeError, ValueError):
        return stream


def matches_preamble_kernel(low_name, preamble):
    if preamble == "all_reduce":
        return "allreduce" in low_name
    if preamble == "all_gather":
        return "allgather" in low_name
    return False


def parse_trace_stats(trace_path, preamble, total_all2all_iters, alltoall_matcher=None, stream_filter=None):
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    if alltoall_matcher is None:
        alltoall_matcher = matches_all_to_all_kernel
    if stream_filter is not None:
        stream_filter = set(stream_filter)

    alltoall_us = 0.0
    preamble_us = 0.0
    alltoall_count = 0
    preamble_count = 0
    matched_alltoall_names = set()
    matched_preamble_names = set()
    matched_alltoall_streams = set()
    matched_preamble_streams = set()
    alltoall_durations = []
    preamble_durations = []
    observed_cuda_kernel_names = []
    observed_stream_ids = set()

    for evt in trace.get("traceEvents", []):
        if evt.get("ph") != "X":
            continue
        name = evt.get("name", "")
        dur = float(evt.get("dur", 0.0))
        low = name.lower()
        stream_id = kernel_stream_id(evt)

        if evt.get("cat") == "kernel" and name and name not in observed_cuda_kernel_names:
            observed_cuda_kernel_names.append(name)
        if evt.get("cat") == "kernel" and stream_id is not None:
            observed_stream_ids.add(stream_id)
        if evt.get("cat") != "kernel":
            continue
        if stream_filter is not None and stream_id not in stream_filter:
            continue

        if alltoall_matcher(low):
            alltoall_us += dur
            alltoall_count += 1
            matched_alltoall_names.add(name)
            alltoall_durations.append(dur)
            if stream_id is not None:
                matched_alltoall_streams.add(stream_id)

        if matches_preamble_kernel(low, preamble):
            preamble_us += dur
            preamble_count += 1
            matched_preamble_names.add(name)
            preamble_durations.append(dur)
            if stream_id is not None:
                matched_preamble_streams.add(stream_id)

    if preamble != "none":
        comm_region_durations = [
            preamble_durations[i] + alltoall_durations[i]
            for i in range(min(len(preamble_durations), len(alltoall_durations)))
        ]
    else:
        comm_region_durations = list(alltoall_durations)

    stats = {
        "all2all": {
            "trace_avg_us": alltoall_us / max(total_all2all_iters, 1),
            "trace_p50_us": summarize_values(alltoall_durations)["p50_us"],
            "kernel_count": alltoall_count,
            "matched_kernel_names": sorted(matched_alltoall_names),
            "matched_stream_ids": sorted(matched_alltoall_streams),
        },
        "comm_region": {
            "trace_avg_us": (preamble_us + alltoall_us) / max(total_all2all_iters, 1),
            "trace_p50_us": summarize_values(comm_region_durations)["p50_us"],
            "kernel_count": alltoall_count + preamble_count,
            "matched_stream_ids": sorted(matched_alltoall_streams | matched_preamble_streams),
        },
        "observed_cuda_kernel_names": observed_cuda_kernel_names[:32],
        "observed_stream_ids": sorted(observed_stream_ids),
    }
    if preamble != "none":
        stats["preamble"] = {
            "trace_avg_us": preamble_us / max(total_all2all_iters, 1),
            "trace_p50_us": summarize_values(preamble_durations)["p50_us"],
            "kernel_count": preamble_count,
            "matched_kernel_names": sorted(matched_preamble_names),
            "matched_stream_ids": sorted(matched_preamble_streams),
        }
    else:
        stats["preamble"] = {
            "trace_avg_us": 0.0,
            "trace_p50_us": 0.0,
            "kernel_count": 0,
            "matched_kernel_names": [],
            "matched_stream_ids": [],
        }
    stats["matched_target_stream_ids"] = sorted(matched_alltoall_streams | matched_preamble_streams)
    return stats


def profiler_benchmark(
    runner,
    graph,
    mode,
    profile_iters,
    trace_path,
    device=None,
    group=None,
    before_iter=None,
    alltoall_matcher=None,
    stream_filter=None,
    replay_stream=None,
):
    if device is None:
        device = runner.device
    if group is None:
        group = runner.group
    if profile_iters <= 0:
        raise ValueError("profile_iters must be positive")
    dist.barrier(group=group)
    torch.cuda.synchronize(device)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    warmup_steps = 3
    schedule = torch.profiler.schedule(wait=0, warmup=warmup_steps, active=profile_iters, repeat=1)
    with torch.profiler.profile(activities=activities, record_shapes=False, schedule=schedule) as prof:
        for _ in range(profile_iters + warmup_steps):
            if before_iter is not None:
                before_iter()
            if mode == "graph":
                if replay_stream is None:
                    graph.replay()
                else:
                    with torch.cuda.stream(replay_stream):
                        graph.replay()
            else:
                runner.run_once()
            torch.cuda.synchronize(device)
            prof.step()
    dist.barrier(group=group)
    prof.export_chrome_trace(trace_path)
    total_all2all_iters = profile_iters * runner.inner_iters
    stats = parse_trace_stats(
        trace_path,
        runner.preamble,
        total_all2all_iters,
        alltoall_matcher=alltoall_matcher,
        stream_filter=stream_filter,
    )
    if stats["all2all"]["kernel_count"] == 0:
        raise RuntimeError(
            "Failed to locate target all_to_all kernels in profiler trace "
            f"{trace_path}. Observed CUDA kernels: {stats['observed_cuda_kernel_names']}"
        )
    return stats


def validate_runner(runner, expected, group, device, payload_name, cp_size, cp_rank):
    runner.run_once()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    actual = runner.output.clone()
    ok = torch.equal(actual, expected)
    ok_tensor = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN, group=group)
    if ok_tensor.item() != 1:
        mismatch = (actual.float() - expected.float()).abs().max().item()
        raise RuntimeError(
            f"Validation failed for payload={payload_name}, cp_size={cp_size}, cp_rank={cp_rank}, max_diff={mismatch}"
        )


def gather_rank_meta(meta, group):
    gathered = [None for _ in range(dist.get_world_size(group=group))]
    dist.all_gather_object(gathered, meta, group=group)
    return gathered


def active_aggregate(rank_items, key):
    values = [item[key]["mean_us"] for item in rank_items]
    return {
        "mean_of_means_us": float(sum(values) / len(values)) if values else 0.0,
        "slowest_mean_us": float(max(values)) if values else 0.0,
        "fastest_mean_us": float(min(values)) if values else 0.0,
    }


def zero_payload_summary(payload, cp_size):
    return {
        "payload": payload.name,
        "traffic_pattern": payload.traffic_pattern,
        "feature_dim": payload.feature_dim,
        "row_bytes": 0,
        "total_dispatch_bytes": 0,
        "active_ranks": [0],
        "ranks": [
            {
                "global_rank": 0,
                "cp_rank": 0,
                "wall_summary_us": summarize_values([0.0]),
                "all2all_summary_us": summarize_values([0.0]),
                "trace_paths": [],
            }
        ],
        "wall_active_summary": {
            "mean_of_means_us": 0.0,
            "slowest_mean_us": 0.0,
            "fastest_mean_us": 0.0,
        },
        "all2all_active_summary": {
            "mean_of_means_us": 0.0,
            "slowest_mean_us": 0.0,
            "fastest_mean_us": 0.0,
        },
    }


def benchmark_cp_size(args, group, global_rank, device, cp_size, dtype, trace_root):
    cp_rank = dist.get_rank(group=group)
    inner_iters = args.graph_inner_iters if args.mode == "graph" else 1
    specs = payload_specs(args.num_heads, args.head_dim, args.v_head_dim, cp_size)
    results = {}

    if cp_size == 1:
        for payload in specs:
            results[payload.name] = zero_payload_summary(payload, cp_size)
        return results

    itemsize = torch.tensor([], dtype=dtype).element_size()
    common_buffer_size = AllToAllIntraLLBuffer.get_buffer_size_hint(
        cp_size,
        args.max_num_seqs,
        shared_msg_size(args.num_heads, args.head_dim),
        itemsize,
    )

    for payload in specs:
        buffer = AllToAllIntraLLBuffer(
            payload.max_dispatch_per_msg,
            args.max_num_seqs,
            cp_rank,
            cp_size,
            common_buffer_size,
        )
        buffer.connect_full_mesh(group)

        if payload.name == "Q":
            x, mask, expected = build_q_setup(
                buffer,
                cp_rank,
                cp_size,
                args.max_num_seqs,
                args.num_requests,
                dtype,
                device,
                payload.feature_dim,
            )
        elif payload.name == "Res":
            x, mask, expected = build_reduce_setup(
                buffer,
                cp_rank,
                cp_size,
                args.max_num_seqs,
                args.num_requests,
                dtype,
                device,
                payload.feature_dim,
                value_fn=res_seq_value,
            )
        else:
            x, mask, expected = build_reduce_setup(
                buffer,
                cp_rank,
                cp_size,
                args.max_num_seqs,
                args.num_requests,
                dtype,
                device,
                payload.feature_dim,
                value_fn=lse_seq_value,
            )

        runner = LLRunner(
            group=group,
            preamble=args.preamble,
            device=device,
            inner_iters=inner_iters,
            buffer=buffer,
            x=x,
            mask=mask,
            is_transpose=payload.is_transpose,
        )
        validate_runner(runner, expected, group, device, payload.name, cp_size, cp_rank)

        if args.mode == "graph":
            graph = capture_graph(runner, args.warmup, group, device)
        else:
            graph = None
            for _ in range(args.warmup):
                runner.run_once()
            torch.cuda.synchronize(device)
            dist.barrier(group=group)

        wall_us_samples = []
        trace_us_samples = []
        trace_paths = []
        cp_trace_dir = trace_root / f"cp{cp_size}"
        cp_trace_dir.mkdir(parents=True, exist_ok=True)

        for repeat_idx in range(args.repeats):
            wall_us = walltime_benchmark(runner, graph, args.mode, args.iters, device, group)
            wall_us_samples.append(wall_us)

            if args.profile_iters > 0:
                trace_path = cp_trace_dir / (
                    f"{payload.name.lower()}_globalrank{global_rank}_cp{cp_size}_{args.mode}_{args.preamble}_repeat{repeat_idx}.json"
                )
                prof_stats = profiler_benchmark(runner, graph, args.mode, args.profile_iters, str(trace_path))
                trace_us_samples.append(prof_stats["all2all"]["trace_p50_us"])
                trace_paths.append(str(trace_path.resolve()))

        row_bytes = payload.feature_dim * torch.tensor([], dtype=dtype).element_size()
        logical_dispatches = int(mask.sum().item())
        meta = {
            "global_rank": global_rank,
            "cp_rank": cp_rank,
            "payload": payload.name,
            "feature_dim": payload.feature_dim,
            "row_bytes": row_bytes,
            "logical_dispatches": logical_dispatches,
            "traffic_pattern": payload.traffic_pattern,
            "wall_summary_us": summarize_values(wall_us_samples),
            "all2all_summary_us": summarize_values(trace_us_samples),
            "trace_paths": trace_paths,
        }
        gathered = gather_rank_meta(meta, group)

        if cp_rank == 0:
            results[payload.name] = {
                "payload": payload.name,
                "traffic_pattern": payload.traffic_pattern,
                "feature_dim": payload.feature_dim,
                "row_bytes": row_bytes,
                "total_dispatch_bytes": sum(item["logical_dispatches"] for item in gathered) * row_bytes,
                "active_ranks": [item["global_rank"] for item in gathered],
                "ranks": gathered,
                "wall_active_summary": active_aggregate(gathered, "wall_summary_us"),
                "all2all_active_summary": active_aggregate(gathered, "all2all_summary_us"),
            }

    return results


def estimated_total(results, summary_key):
    total = 0.0
    for payload_name in ("Q", "Res", "Lse"):
        total += results[payload_name][summary_key]["slowest_mean_us"]
    return total


def main():
    args = parse_args()
    if args.max_num_seqs <= 0:
        raise ValueError("--max-num-seqs must be positive")
    if args.num_requests <= 0:
        raise ValueError("--num-requests must be positive")
    if args.max_num_seqs < args.num_requests:
        args.max_num_seqs = args.num_requests

    dtype = get_dtype(args.dtype)
    cp_sizes = parse_cp_sizes(args.cp_sizes)
    global_rank, world_size, device = init_dist()
    max_cp_size = max(cp_sizes)
    if max_cp_size > world_size:
        raise ValueError(f"Requested max CP size {max_cp_size}, but world_size is only {world_size}")

    trace_root = Path(args.trace_dir)
    if global_rank == 0:
        trace_root.mkdir(parents=True, exist_ok=True)
        Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    for cp_size in cp_sizes:
        group = make_group(cp_size)
        active = global_rank < cp_size

        if active:
            payload_results = benchmark_cp_size(args, group, global_rank, device, cp_size, dtype, trace_root)
            if dist.get_rank(group=group) == 0:
                all_results.append(
                    {
                        "cp_size": cp_size,
                        "mode": args.mode,
                        "preamble": args.preamble,
                        "dtype": args.dtype,
                        "num_heads": args.num_heads,
                        "head_dim": args.head_dim,
                        "v_head_dim": args.v_head_dim,
                        "batch_size": 1,
                        "num_requests": args.num_requests,
                        "max_num_seqs": args.max_num_seqs,
                        "payloads": payload_results,
                        "estimated_total_wall_slowest_us": estimated_total(payload_results, "wall_active_summary"),
                        "estimated_total_all2all_slowest_us": estimated_total(
                            payload_results, "all2all_active_summary"
                        ),
                    }
                )

        dist.barrier()

    if global_rank == 0:
        summary = {
            "description": "Single-request DeepSeek-V3 MLA communication estimate across CP sizes",
            "master_rank": MASTER_RANK,
            "cp_sizes": cp_sizes,
            "mode": args.mode,
            "preamble": args.preamble,
            "dtype": args.dtype,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
            "v_head_dim": args.v_head_dim,
            "batch_size": 1,
            "num_requests": args.num_requests,
            "max_num_seqs": args.max_num_seqs,
            "results": all_results,
        }
        with open(args.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("Saved summary to", Path(args.summary_path).resolve())
        for item in all_results:
            print(
                f"cp_size={item['cp_size']} | "
                f"q_all2all_slowest_us={item['payloads']['Q']['all2all_active_summary']['slowest_mean_us']:.2f} | "
                f"res_all2all_slowest_us={item['payloads']['Res']['all2all_active_summary']['slowest_mean_us']:.2f} | "
                f"lse_all2all_slowest_us={item['payloads']['Lse']['all2all_active_summary']['slowest_mean_us']:.2f} | "
                f"estimated_total_all2all_slowest_us={item['estimated_total_all2all_slowest_us']:.2f}"
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
