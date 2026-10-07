import argparse
import csv
import json
from pathlib import Path

import torch
import torch.distributed as dist

from benchmark_mla_cp_comm import (
    BaseRunner,
    MASTER_RANK,
    active_aggregate,
    gather_rank_meta,
    get_dtype,
    init_dist,
    make_group,
    parse_cp_sizes,
    parse_trace_stats,
    payload_specs,
    profiler_benchmark,
    summarize_values,
)

try:
    import dlslime
    import dlslime._slime_c as slime_c
    from dlslime import AllToAllBuffer, KernelImpl
except ImportError:
    dlslime = None
    slime_c = None
    AllToAllBuffer = None
    KernelImpl = None


BACKGROUND_BASE_VALUES = {
    "Q": 3000.0,
    "Res": 4000.0,
    "Lse": 5000.0,
}

TARGET_BASE_VALUES = {
    "Q": 1000.0,
    "Res": 2000.0,
    "Lse": 2500.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure single-request DeepSeek-V3 MLA communication with "
            "DLSlime AllToAllBuffer across CP sizes, both idle and under background load."
        )
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
        "--target-bs-per-rank",
        type=str,
        default="",
        help=(
            "Optional comma-separated per-rank local batch sizes for the target all_to_all. "
            "If set, the target benchmark switches from request-level MLA semantics to a "
            "generic remote all_to_all benchmark. A single integer means uniform bs on all active ranks."
        ),
    )
    parser.add_argument(
        "--target-fanout-per-rank",
        type=str,
        default="",
        help=(
            "Optional comma-separated per-rank remote fanout counts for the target all_to_all. "
            "A single integer means uniform fanout on all active ranks. Self-send is always ignored."
        ),
    )
    parser.add_argument("--mode", type=str, default="graph", choices=["eager", "graph"])
    parser.add_argument(
        "--preamble",
        type=str,
        default="all_reduce",
        choices=["none", "all_reduce", "all_gather"],
        help="Optional sync collective before the target all_to_all call.",
    )
    parser.add_argument(
        "--dlslime-impl",
        type=str,
        default="basic",
        choices=["basic", "tma"],
        help="AllToAllBuffer kernel implementation",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50, help="Measured target replays per repeat")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--profile-iters", type=int, default=10)
    parser.add_argument(
        "--graph-inner-iters",
        type=int,
        default=20,
        help="How many target all_to_all launches are captured in one target graph replay",
    )
    parser.add_argument(
        "--bg-payload",
        type=str,
        default="Res",
        choices=["Q", "Res", "Lse"],
        help="Background regular all-to-all payload shape",
    )
    parser.add_argument(
        "--bg-bs-per-rank",
        type=int,
        default=8,
        help="Background message rows per destination rank on each active GPU",
    )
    parser.add_argument(
        "--bg-bs-per-rank-list",
        type=str,
        default="",
        help=(
            "Optional comma-separated per-rank local batch sizes for the background all_to_all. "
            "If set, overrides --bg-bs-per-rank. A single integer means uniform bs on all active ranks."
        ),
    )
    parser.add_argument(
        "--bg-fanout-per-rank",
        type=str,
        default="",
        help=(
            "Optional comma-separated per-rank remote fanout counts for the background all_to_all. "
            "A single integer means uniform fanout on all active ranks. Self-send is always ignored."
        ),
    )
    parser.add_argument(
        "--bg-inner-iters",
        type=int,
        default=1,
        help="Background all_to_all launches per replay when mode=graph",
    )
    parser.add_argument(
        "--bg-pre-replays",
        type=int,
        default=8,
        help="How many background replays to enqueue before each target replay",
    )
    parser.add_argument(
        "--bg-preamble",
        type=str,
        default="none",
        choices=["none", "all_reduce", "all_gather"],
        help="Optional sync collective before the background all-to-all",
    )
    parser.add_argument("--trace-dir", type=str, default="results_dlslime_alltoall/traces")
    parser.add_argument("--summary-path", type=str, default="results_dlslime_alltoall/summary.json")
    parser.add_argument("--csv-path", type=str, default="results_dlslime_alltoall/comparison.csv")
    return parser.parse_args()


def ensure_dlslime():
    if AllToAllBuffer is None or KernelImpl is None or slime_c is None:
        raise ImportError("Failed to import AllToAllBuffer / KernelImpl from dlslime")
    if not getattr(slime_c, "_BUILD_INTRA_OPS", False):
        raise RuntimeError("This dlslime build does not expose intra ops (AllToAllBuffer)")


def get_kernel_impl(impl_name):
    ensure_dlslime()
    return {
        "basic": KernelImpl.Basic,
        "tma": KernelImpl.TMA,
    }[impl_name]


def make_row(fill_value, feature_dim, dtype, device):
    return torch.full((feature_dim,), fill_value=fill_value, dtype=dtype, device=device)


def parse_positive_int_list(text, arg_name):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"{arg_name} entries must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{arg_name} must provide at least one positive integer")
    return values


def parse_nonnegative_int_list(text, arg_name):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 0:
            raise ValueError(f"{arg_name} entries must be non-negative, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{arg_name} must provide at least one non-negative integer")
    return values


def resolve_bs_vector(text, default_bs, cp_size, arg_name):
    if text.strip():
        values = parse_positive_int_list(text, arg_name)
        if len(values) == 1:
            return values * cp_size
        if len(values) < cp_size:
            raise ValueError(
                f"{arg_name} must provide either 1 value or at least {cp_size} values, got {len(values)}"
            )
        return values[:cp_size]
    return [default_bs] * cp_size


def resolve_fanout_vector(text, cp_size, arg_name):
    default_fanout = max(cp_size - 1, 0)
    if text.strip():
        values = parse_nonnegative_int_list(text, arg_name)
        if len(values) == 1:
            values = values * cp_size
        elif len(values) < cp_size:
            raise ValueError(
                f"{arg_name} must provide either 1 value or at least {cp_size} values, got {len(values)}"
            )
        else:
            values = values[:cp_size]
        max_remote = max(cp_size - 1, 0)
        for value in values:
            if value > max_remote:
                raise ValueError(f"{arg_name} entries must be <= {max_remote}, got {value}")
        return values
    return [default_fanout] * cp_size


def format_bs_vector(values):
    return ",".join(str(v) for v in values)


def q_payload_value():
    return 1.0


def q_seq_value(seq_idx):
    return 1.0 + float(seq_idx)


def res_seq_value(cp_rank, seq_idx):
    return 128.0 + float(cp_rank * 16 + seq_idx)


def lse_seq_value(cp_rank, seq_idx):
    return float(cp_rank * 16 + seq_idx)


def generic_seq_value(base_value, cp_rank, seq_idx):
    return base_value + float(cp_rank * 1000 + seq_idx)


def remote_destinations_for_row(cp_rank, cp_size, row_idx, fanout):
    remote_ranks = [rank for rank in range(cp_size) if rank != cp_rank]
    if fanout <= 0 or not remote_ranks:
        return []
    fanout = min(fanout, len(remote_ranks))
    start = row_idx % len(remote_ranks)
    return [remote_ranks[(start + offset) % len(remote_ranks)] for offset in range(fanout)]


def background_feature_dim(bg_payload_name, num_heads, head_dim, v_head_dim):
    if bg_payload_name == "Q":
        return num_heads * head_dim
    if bg_payload_name == "Res":
        return num_heads * v_head_dim
    return num_heads


def create_buffer(cp_rank, cp_size, max_batch_size, feature_dim, dtype, group):
    itemsize = torch.tensor([], dtype=dtype).element_size()
    buffer_size_bytes = max_batch_size * cp_size * feature_dim * itemsize
    buffer = AllToAllBuffer(cp_rank, cp_size, max_batch_size, buffer_size_bytes)
    my_handle_info = buffer.get_ipc_handle_info()
    all_handle_infos = [None for _ in range(cp_size)]
    dist.all_gather_object(all_handle_infos, my_handle_info, group=group)
    buffer.connect_full_mesh(all_handle_infos)
    return buffer


def matches_buffer_all_to_all_kernel(low_name):
    return (
        ("alltoall" in low_name or "all_to_all" in low_name)
        and ("intranode" in low_name or "dlslime" in low_name or "slime" in low_name)
    )


def build_q_setup(cp_rank, cp_size, num_requests, dtype, device, feature_dim):
    x = torch.zeros((num_requests, feature_dim), dtype=dtype, device=device)
    mask = torch.zeros((num_requests, cp_size), dtype=torch.int32, device=device)

    if cp_rank == MASTER_RANK:
        for seq_idx in range(num_requests):
            x[seq_idx].copy_(make_row(q_seq_value(seq_idx), feature_dim, dtype, device))
            for dst in range(cp_size):
                if dst == MASTER_RANK:
                    continue
                mask[seq_idx, dst] = 1

    expected = torch.zeros((cp_size, num_requests, feature_dim), dtype=dtype, device=device)
    if cp_rank != MASTER_RANK:
        for seq_idx in range(num_requests):
            expected[MASTER_RANK, seq_idx].copy_(
                make_row(q_seq_value(seq_idx), feature_dim, dtype, device)
            )
    return x, mask, expected, num_requests


def build_reduce_setup(cp_rank, cp_size, num_requests, dtype, device, feature_dim, value_fn):
    x = torch.zeros((num_requests, feature_dim), dtype=dtype, device=device)
    mask = torch.zeros((num_requests, cp_size), dtype=torch.int32, device=device)

    if cp_rank != MASTER_RANK:
        for seq_idx in range(num_requests):
            row = make_row(value_fn(cp_rank, seq_idx), feature_dim, dtype, device)
            x[seq_idx].copy_(row)
            mask[seq_idx, MASTER_RANK] = 1

    expected = torch.zeros((cp_size, num_requests, feature_dim), dtype=dtype, device=device)
    if cp_rank == MASTER_RANK:
        for src in range(cp_size):
            if src == MASTER_RANK:
                continue
            for seq_idx in range(num_requests):
                expected[src, seq_idx].copy_(make_row(value_fn(src, seq_idx), feature_dim, dtype, device))
    return x, mask, expected, num_requests


def build_remote_alltoall_setup(cp_rank, cp_size, bs_vector, fanout_vector, dtype, device, feature_dim, base_value):
    local_bs = bs_vector[cp_rank]
    max_bs = max(bs_vector)
    local_fanout = fanout_vector[cp_rank]

    x = torch.zeros((local_bs, feature_dim), dtype=dtype, device=device)
    mask = torch.zeros((local_bs, cp_size), dtype=torch.int32, device=device)

    for seq_idx in range(local_bs):
        x[seq_idx].copy_(make_row(generic_seq_value(base_value, cp_rank, seq_idx), feature_dim, dtype, device))
        for dst in remote_destinations_for_row(cp_rank, cp_size, seq_idx, local_fanout):
            mask[seq_idx, dst] = 1

    expected = torch.zeros((cp_size, max_bs, feature_dim), dtype=dtype, device=device)
    for src in range(cp_size):
        for seq_idx in range(bs_vector[src]):
            if cp_rank in remote_destinations_for_row(src, cp_size, seq_idx, fanout_vector[src]):
                expected[src, seq_idx].copy_(
                    make_row(generic_seq_value(base_value, src, seq_idx), feature_dim, dtype, device)
                )
    return x, mask, expected, max_bs


def build_background_dense_setup(cp_rank, cp_size, bg_bs_per_rank, dtype, device, feature_dim, base_value):
    padded_rows = cp_size * bg_bs_per_rank
    x = torch.zeros((padded_rows, feature_dim), dtype=dtype, device=device)
    mask = torch.zeros((padded_rows, cp_size), dtype=torch.int32, device=device)

    for dst in range(cp_size):
        for seq in range(bg_bs_per_rank):
            row_idx = dst * bg_bs_per_rank + seq
            value = base_value + float(cp_rank * 100 + dst * 10 + seq)
            x[row_idx].fill_(value)
            mask[row_idx, dst] = 1

    expected = torch.zeros((cp_size, padded_rows, feature_dim), dtype=dtype, device=device)
    base_offset = cp_rank * bg_bs_per_rank
    for src in range(cp_size):
        for seq in range(bg_bs_per_rank):
            row_idx = base_offset + seq
            value = base_value + float(src * 100 + cp_rank * 10 + seq)
            expected[src, row_idx].fill_(value)
    return x, mask, expected, padded_rows


def build_background_setup(
    cp_rank,
    cp_size,
    bg_bs_vector,
    bg_fanout_vector,
    dtype,
    device,
    feature_dim,
    base_value,
    use_vector_mode,
):
    if use_vector_mode:
        return build_remote_alltoall_setup(
            cp_rank,
            cp_size,
            bg_bs_vector,
            bg_fanout_vector,
            dtype,
            device,
            feature_dim,
            base_value,
        )
    return build_background_dense_setup(cp_rank, cp_size, bg_bs_vector[cp_rank], dtype, device, feature_dim, base_value)


def setup_target_payload(
    payload,
    cp_rank,
    cp_size,
    num_requests,
    target_bs_vector,
    target_fanout_vector,
    use_vector_mode,
    dtype,
    device,
):
    if use_vector_mode:
        return build_remote_alltoall_setup(
            cp_rank,
            cp_size,
            target_bs_vector,
            target_fanout_vector,
            dtype,
            device,
            payload.feature_dim,
            TARGET_BASE_VALUES[payload.name],
        )
    if payload.name == "Q":
        return build_q_setup(cp_rank, cp_size, num_requests, dtype, device, payload.feature_dim)
    if payload.name == "Res":
        return build_reduce_setup(cp_rank, cp_size, num_requests, dtype, device, payload.feature_dim, value_fn=res_seq_value)
    return build_reduce_setup(cp_rank, cp_size, num_requests, dtype, device, payload.feature_dim, value_fn=lse_seq_value)


class BufferRunner(BaseRunner):
    def __init__(self, group, preamble, device, inner_iters, buffer, x, mask, kernel_impl):
        super().__init__(group=group, preamble=preamble, device=device, inner_iters=inner_iters)
        self.buffer = buffer
        self.x = x
        self.mask = mask
        self.kernel_impl = kernel_impl
        self.output = None

    def run_once(self):
        for _ in range(self.inner_iters):
            self.preamble_op()
            self.output = self.buffer.all_to_all(
                self.x,
                impl=self.kernel_impl,
                is_transpose=True,
                mask=self.mask,
            )


def capture_graph_on_stream(runner, warmup, group, device, stream):
    current_stream = torch.cuda.current_stream(device)
    stream.wait_stream(current_stream)
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            runner.run_once()
    stream.synchronize()
    dist.barrier(group=group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        runner.run_once()
    stream.synchronize()
    return graph


def replay_runner(runner, graph, mode, stream):
    if mode == "graph":
        graph.replay()
        return
    with torch.cuda.stream(stream):
        runner.run_once()


def target_event_samples(runner, graph, mode, stream, iters, repeats, group, device, before_target=None):
    per_op_scale = max(runner.inner_iters, 1)
    samples = []
    for _ in range(repeats):
        dist.barrier(group=group)
        torch.cuda.synchronize(device)
        event_pairs = []
        for _ in range(iters):
            if before_target is not None:
                before_target()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            replay_runner(runner, graph, mode, stream)
            end.record(stream)
            event_pairs.append((start, end))

        for _, end in event_pairs:
            end.synchronize()
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

        for start, end in event_pairs:
            samples.append(start.elapsed_time(end) * 1000.0 / per_op_scale)
    return samples


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


def validate_under_load(
    target_runner,
    target_graph,
    target_expected,
    target_stream,
    bg_launch,
    mode,
    group,
    device,
    payload_name,
    cp_size,
    cp_rank,
):
    dist.barrier(group=group)
    torch.cuda.synchronize(device)
    bg_launch()
    replay_runner(target_runner, target_graph, mode, target_stream)
    torch.cuda.synchronize(device)
    dist.barrier(group=group)

    actual = target_runner.output.clone()
    ok = torch.equal(actual, target_expected)
    ok_tensor = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN, group=group)
    if ok_tensor.item() != 1:
        mismatch = (actual.float() - target_expected.float()).abs().max().item()
        raise RuntimeError(
            f"Under-load validation failed for payload={payload_name}, cp_size={cp_size}, "
            f"cp_rank={cp_rank}, max_diff={mismatch}"
        )


def zero_payload_summary(payload, cp_size, bg_payload, bg_bs_vector):
    return {
        "payload": payload.name,
        "feature_dim": payload.feature_dim,
        "row_bytes": 0,
        "active_ranks": [0],
        "background": {
            "payload": bg_payload,
            "bs_per_rank": list(bg_bs_vector),
            "traffic_pattern": "dense_all_to_all",
            "feature_dim": 0,
            "bg_inner_iters": 0,
            "bg_pre_replays": 0,
            "bg_preamble": "none",
        },
        "ranks": [
            {
                "global_rank": 0,
                "cp_rank": 0,
                "idle_event_summary_us": summarize_values([0.0]),
                "loaded_event_summary_us": summarize_values([0.0]),
                "idle_preamble_summary_us": summarize_values([0.0]),
                "loaded_preamble_summary_us": summarize_values([0.0]),
                "idle_all2all_summary_us": summarize_values([0.0]),
                "loaded_all2all_summary_us": summarize_values([0.0]),
                "idle_comm_region_summary_us": summarize_values([0.0]),
                "loaded_comm_region_summary_us": summarize_values([0.0]),
            }
        ],
        "idle_event_active_summary": {
            "mean_of_means_us": 0.0,
            "slowest_mean_us": 0.0,
            "fastest_mean_us": 0.0,
        },
        "loaded_event_active_summary": {
            "mean_of_means_us": 0.0,
            "slowest_mean_us": 0.0,
            "fastest_mean_us": 0.0,
        },
        "idle_preamble_active_summary": {
            "mean_of_means_us": 0.0,
            "slowest_mean_us": 0.0,
            "fastest_mean_us": 0.0,
        },
        "loaded_preamble_active_summary": {
            "mean_of_means_us": 0.0,
            "slowest_mean_us": 0.0,
            "fastest_mean_us": 0.0,
        },
        "idle_all2all_active_summary": {
            "mean_of_means_us": 0.0,
            "slowest_mean_us": 0.0,
            "fastest_mean_us": 0.0,
        },
        "loaded_all2all_active_summary": {
            "mean_of_means_us": 0.0,
            "slowest_mean_us": 0.0,
            "fastest_mean_us": 0.0,
        },
        "idle_comm_region_active_summary": {
            "mean_of_means_us": 0.0,
            "slowest_mean_us": 0.0,
            "fastest_mean_us": 0.0,
        },
        "loaded_comm_region_active_summary": {
            "mean_of_means_us": 0.0,
            "slowest_mean_us": 0.0,
            "fastest_mean_us": 0.0,
        },
        "delta_slowest_mean_us": 0.0,
        "slowdown_ratio": 1.0,
    }


def estimate_total(payload_results, key):
    total = 0.0
    for payload_name in ("Q", "Res", "Lse"):
        total += payload_results[payload_name][key]["slowest_mean_us"]
    return total


def benchmark_cp_size(args, group, global_rank, device, cp_size, dtype, kernel_impl, trace_root):
    cp_rank = dist.get_rank(group=group)
    target_specs = payload_specs(args.num_heads, args.head_dim, args.v_head_dim, cp_size)
    results = {}
    use_target_vector_mode = bool(args.target_bs_per_rank.strip())
    use_bg_vector_mode = bool(args.bg_bs_per_rank_list.strip())
    target_bs_vector = resolve_bs_vector(args.target_bs_per_rank, args.num_requests, cp_size, "--target-bs-per-rank")
    bg_bs_vector = resolve_bs_vector(args.bg_bs_per_rank_list, args.bg_bs_per_rank, cp_size, "--bg-bs-per-rank-list")
    target_fanout_vector = resolve_fanout_vector(args.target_fanout_per_rank, cp_size, "--target-fanout-per-rank")
    bg_fanout_vector = resolve_fanout_vector(args.bg_fanout_per_rank, cp_size, "--bg-fanout-per-rank")

    if cp_size == 1:
        for payload in target_specs:
            results[payload.name] = zero_payload_summary(payload, cp_size, args.bg_payload, bg_bs_vector)
        return results

    bg_feature_dim = background_feature_dim(args.bg_payload, args.num_heads, args.head_dim, args.v_head_dim)
    target_stream = torch.cuda.Stream(device=device)
    bg_stream = torch.cuda.Stream(device=device)

    for payload in target_specs:
        target_x, target_mask, target_expected, target_rows = setup_target_payload(
            payload,
            cp_rank,
            cp_size,
            args.num_requests,
            target_bs_vector,
            target_fanout_vector,
            use_target_vector_mode,
            dtype,
            device,
        )
        target_buffer = create_buffer(cp_rank, cp_size, target_rows, payload.feature_dim, dtype, group)
        target_runner = BufferRunner(
            group=group,
            preamble=args.preamble,
            device=device,
            inner_iters=args.graph_inner_iters if args.mode == "graph" else 1,
            buffer=target_buffer,
            x=target_x,
            mask=target_mask,
            kernel_impl=kernel_impl,
        )
        validate_runner(target_runner, target_expected, group, device, payload.name, cp_size, cp_rank)
        if args.mode == "graph":
            target_graph = capture_graph_on_stream(target_runner, args.warmup, group, device, target_stream)
        else:
            target_graph = None
            with torch.cuda.stream(target_stream):
                for _ in range(args.warmup):
                    target_runner.run_once()
            target_stream.synchronize()
            dist.barrier(group=group)

        bg_x, bg_mask, bg_expected, bg_rows = build_background_setup(
            cp_rank,
            cp_size,
            bg_bs_vector,
            bg_fanout_vector,
            dtype,
            device,
            bg_feature_dim,
            BACKGROUND_BASE_VALUES[args.bg_payload],
            use_bg_vector_mode,
        )
        bg_buffer = create_buffer(cp_rank, cp_size, bg_rows, bg_feature_dim, dtype, group)
        bg_runner = BufferRunner(
            group=group,
            preamble=args.bg_preamble,
            device=device,
            inner_iters=args.bg_inner_iters if args.mode == "graph" else 1,
            buffer=bg_buffer,
            x=bg_x,
            mask=bg_mask,
            kernel_impl=kernel_impl,
        )
        validate_runner(bg_runner, bg_expected, group, device, f"BG_{args.bg_payload}", cp_size, cp_rank)
        if args.mode == "graph":
            bg_graph = capture_graph_on_stream(bg_runner, args.warmup, group, device, bg_stream)
        else:
            bg_graph = None
            with torch.cuda.stream(bg_stream):
                for _ in range(args.warmup):
                    bg_runner.run_once()
            bg_stream.synchronize()
            dist.barrier(group=group)

        def launch_background():
            for _ in range(args.bg_pre_replays):
                replay_runner(bg_runner, bg_graph, args.mode, bg_stream)

        validate_under_load(
            target_runner,
            target_graph,
            target_expected,
            target_stream,
            launch_background,
            args.mode,
            group,
            device,
            payload.name,
            cp_size,
            cp_rank,
        )

        idle_event_samples = target_event_samples(
            target_runner,
            target_graph,
            args.mode,
            target_stream,
            args.iters,
            args.repeats,
            group,
            device,
        )
        loaded_event_samples = target_event_samples(
            target_runner,
            target_graph,
            args.mode,
            target_stream,
            args.iters,
            args.repeats,
            group,
            device,
            before_target=launch_background,
        )

        idle_preamble_trace_samples = []
        loaded_preamble_trace_samples = []
        idle_all2all_trace_samples = []
        loaded_all2all_trace_samples = []
        idle_comm_region_trace_samples = []
        loaded_comm_region_trace_samples = []
        target_stream_ids = None
        cp_trace_dir = trace_root / f"cp{cp_size}"
        cp_trace_dir.mkdir(parents=True, exist_ok=True)
        for repeat_idx in range(args.repeats):
            if args.profile_iters <= 0:
                break
            idle_trace_path = cp_trace_dir / (
                f"{payload.name.lower()}_globalrank{global_rank}_cp{cp_size}_{args.mode}_{args.preamble}_idle_repeat{repeat_idx}.json"
            )
            idle_prof_stats = profiler_benchmark(
                target_runner,
                target_graph,
                args.mode,
                args.profile_iters,
                str(idle_trace_path),
                device=device,
                group=group,
                alltoall_matcher=matches_buffer_all_to_all_kernel,
            )
            target_stream_ids = idle_prof_stats["all2all"]["matched_stream_ids"]
            if target_stream_ids:
                idle_prof_stats = parse_trace_stats(
                    str(idle_trace_path),
                    args.preamble,
                    args.profile_iters * target_runner.inner_iters,
                    alltoall_matcher=matches_buffer_all_to_all_kernel,
                    stream_filter=target_stream_ids,
                )
            idle_preamble_trace_samples.append(idle_prof_stats["preamble"]["trace_p50_us"])
            idle_all2all_trace_samples.append(idle_prof_stats["all2all"]["trace_p50_us"])
            idle_comm_region_trace_samples.append(idle_prof_stats["comm_region"]["trace_p50_us"])

            loaded_trace_path = cp_trace_dir / (
                f"{payload.name.lower()}_globalrank{global_rank}_cp{cp_size}_{args.mode}_{args.preamble}_loaded_repeat{repeat_idx}.json"
            )
            dist.barrier(group=group)
            torch.cuda.synchronize(device)
            launch_background()
            loaded_prof_stats = profiler_benchmark(
                target_runner,
                target_graph,
                args.mode,
                args.profile_iters,
                str(loaded_trace_path),
                device=device,
                group=group,
                alltoall_matcher=matches_buffer_all_to_all_kernel,
                stream_filter=target_stream_ids,
            )
            loaded_preamble_trace_samples.append(loaded_prof_stats["preamble"]["trace_p50_us"])
            loaded_all2all_trace_samples.append(loaded_prof_stats["all2all"]["trace_p50_us"])
            loaded_comm_region_trace_samples.append(loaded_prof_stats["comm_region"]["trace_p50_us"])

        row_bytes = payload.feature_dim * torch.tensor([], dtype=dtype).element_size()
        meta = {
            "global_rank": global_rank,
            "cp_rank": cp_rank,
            "payload": payload.name,
            "feature_dim": payload.feature_dim,
            "row_bytes": row_bytes,
            "idle_event_summary_us": summarize_values(idle_event_samples),
            "loaded_event_summary_us": summarize_values(loaded_event_samples),
            "idle_preamble_summary_us": summarize_values(idle_preamble_trace_samples),
            "loaded_preamble_summary_us": summarize_values(loaded_preamble_trace_samples),
            "idle_all2all_summary_us": summarize_values(idle_all2all_trace_samples),
            "loaded_all2all_summary_us": summarize_values(loaded_all2all_trace_samples),
            "idle_comm_region_summary_us": summarize_values(idle_comm_region_trace_samples),
            "loaded_comm_region_summary_us": summarize_values(loaded_comm_region_trace_samples),
        }
        gathered = gather_rank_meta(meta, group)

        if cp_rank == 0:
            idle_summary = active_aggregate(gathered, "idle_event_summary_us")
            loaded_summary = active_aggregate(gathered, "loaded_event_summary_us")
            idle_preamble_summary = active_aggregate(gathered, "idle_preamble_summary_us")
            loaded_preamble_summary = active_aggregate(gathered, "loaded_preamble_summary_us")
            idle_all2all_summary = active_aggregate(gathered, "idle_all2all_summary_us")
            loaded_all2all_summary = active_aggregate(gathered, "loaded_all2all_summary_us")
            idle_comm_region_summary = active_aggregate(gathered, "idle_comm_region_summary_us")
            loaded_comm_region_summary = active_aggregate(gathered, "loaded_comm_region_summary_us")
            delta = loaded_summary["slowest_mean_us"] - idle_summary["slowest_mean_us"]
            slowdown_ratio = 0.0
            if idle_summary["slowest_mean_us"] > 0.0:
                slowdown_ratio = loaded_summary["slowest_mean_us"] / idle_summary["slowest_mean_us"]
            results[payload.name] = {
                "payload": payload.name,
                "feature_dim": payload.feature_dim,
                "row_bytes": row_bytes,
                "traffic_pattern": "remote_all_to_all" if use_target_vector_mode else payload.traffic_pattern,
                "background": {
                    "payload": args.bg_payload,
                    "bs_per_rank": list(bg_bs_vector),
                    "fanout_per_rank": list(bg_fanout_vector),
                    "traffic_pattern": "remote_all_to_all" if use_bg_vector_mode else "dense_all_to_all",
                    "feature_dim": bg_feature_dim,
                    "bg_inner_iters": args.bg_inner_iters if args.mode == "graph" else 1,
                    "bg_pre_replays": args.bg_pre_replays,
                    "bg_preamble": args.bg_preamble,
                },
                "target": {
                    "mode": "remote_all_to_all" if use_target_vector_mode else "single_request_mla",
                    "bs_per_rank": list(target_bs_vector),
                    "fanout_per_rank": list(target_fanout_vector),
                },
                "active_ranks": [item["global_rank"] for item in gathered],
                "ranks": gathered,
                "idle_event_active_summary": idle_summary,
                "loaded_event_active_summary": loaded_summary,
                "idle_preamble_active_summary": idle_preamble_summary,
                "loaded_preamble_active_summary": loaded_preamble_summary,
                "idle_all2all_active_summary": idle_all2all_summary,
                "loaded_all2all_active_summary": loaded_all2all_summary,
                "idle_comm_region_active_summary": idle_comm_region_summary,
                "loaded_comm_region_active_summary": loaded_comm_region_summary,
                "delta_slowest_mean_us": delta,
                "slowdown_ratio": slowdown_ratio,
            }

    return results


def write_csv(results, csv_path):
    rows = []
    for item in results:
        cp_size = item["cp_size"]
        target_bs_vector = item["target"]["bs_per_rank"]
        target_fanout_vector = item["target"]["fanout_per_rank"]
        bg_bs_vector = item["background"]["bs_per_rank"]
        bg_fanout_vector = item["background"]["fanout_per_rank"]
        for payload_name in ("Q", "Res", "Lse"):
            payload = item["payloads"][payload_name]
            rows.append(
                {
                    "cp_size": cp_size,
                    "target_mode": item["target"]["mode"],
                    "target_bs_per_rank": format_bs_vector(target_bs_vector),
                    "target_bs_min": min(target_bs_vector),
                    "target_bs_max": max(target_bs_vector),
                    "target_bs_sum": sum(target_bs_vector),
                    "target_fanout_per_rank": format_bs_vector(target_fanout_vector),
                    "target_fanout_min": min(target_fanout_vector),
                    "target_fanout_max": max(target_fanout_vector),
                    "target_fanout_sum": sum(target_fanout_vector),
                    "payload": payload_name,
                    "idle_event_slowest_us": payload["idle_event_active_summary"]["slowest_mean_us"],
                    "loaded_event_slowest_us": payload["loaded_event_active_summary"]["slowest_mean_us"],
                    "delta_event_slowest_us": payload["delta_slowest_mean_us"],
                    "slowdown_ratio": payload["slowdown_ratio"],
                    "idle_preamble_kernel_slowest_us": payload["idle_preamble_active_summary"]["slowest_mean_us"],
                    "loaded_preamble_kernel_slowest_us": payload["loaded_preamble_active_summary"]["slowest_mean_us"],
                    "idle_all2all_kernel_slowest_us": payload["idle_all2all_active_summary"]["slowest_mean_us"],
                    "loaded_all2all_kernel_slowest_us": payload["loaded_all2all_active_summary"]["slowest_mean_us"],
                    "idle_comm_region_kernel_slowest_us": payload["idle_comm_region_active_summary"]["slowest_mean_us"],
                    "loaded_comm_region_kernel_slowest_us": payload["loaded_comm_region_active_summary"]["slowest_mean_us"],
                    "delta_comm_region_kernel_slowest_us": payload["loaded_comm_region_active_summary"]["slowest_mean_us"]
                    - payload["idle_comm_region_active_summary"]["slowest_mean_us"],
                    "bg_payload": payload["background"]["payload"],
                    "bg_bs_per_rank": format_bs_vector(bg_bs_vector),
                    "bg_bs_min": min(bg_bs_vector),
                    "bg_bs_max": max(bg_bs_vector),
                    "bg_bs_sum": sum(bg_bs_vector),
                    "bg_fanout_per_rank": format_bs_vector(bg_fanout_vector),
                    "bg_fanout_min": min(bg_fanout_vector),
                    "bg_fanout_max": max(bg_fanout_vector),
                    "bg_fanout_sum": sum(bg_fanout_vector),
                    "bg_inner_iters": payload["background"]["bg_inner_iters"],
                    "bg_pre_replays": payload["background"]["bg_pre_replays"],
                }
            )

        rows.append(
            {
                "cp_size": cp_size,
                "target_mode": item["target"]["mode"],
                "target_bs_per_rank": format_bs_vector(target_bs_vector),
                "target_bs_min": min(target_bs_vector),
                "target_bs_max": max(target_bs_vector),
                "target_bs_sum": sum(target_bs_vector),
                "target_fanout_per_rank": format_bs_vector(target_fanout_vector),
                "target_fanout_min": min(target_fanout_vector),
                "target_fanout_max": max(target_fanout_vector),
                "target_fanout_sum": sum(target_fanout_vector),
                "payload": "TOTAL",
                "idle_event_slowest_us": item["estimated_total_idle_event_slowest_us"],
                "loaded_event_slowest_us": item["estimated_total_loaded_event_slowest_us"],
                "delta_event_slowest_us": item["estimated_total_delta_event_slowest_us"],
                "slowdown_ratio": item["estimated_total_loaded_event_slowest_us"]
                / item["estimated_total_idle_event_slowest_us"]
                if item["estimated_total_idle_event_slowest_us"] > 0.0
                else 1.0,
                "idle_preamble_kernel_slowest_us": item["estimated_total_idle_preamble_slowest_us"],
                "loaded_preamble_kernel_slowest_us": item["estimated_total_loaded_preamble_slowest_us"],
                "idle_all2all_kernel_slowest_us": item["estimated_total_idle_all2all_slowest_us"],
                "loaded_all2all_kernel_slowest_us": item["estimated_total_loaded_all2all_slowest_us"],
                "idle_comm_region_kernel_slowest_us": item["estimated_total_idle_comm_region_slowest_us"],
                "loaded_comm_region_kernel_slowest_us": item["estimated_total_loaded_comm_region_slowest_us"],
                "delta_comm_region_kernel_slowest_us": item["estimated_total_delta_comm_region_slowest_us"],
                "bg_payload": item["background"]["payload"],
                "bg_bs_per_rank": format_bs_vector(bg_bs_vector),
                "bg_bs_min": min(bg_bs_vector),
                "bg_bs_max": max(bg_bs_vector),
                "bg_bs_sum": sum(bg_bs_vector),
                "bg_fanout_per_rank": format_bs_vector(bg_fanout_vector),
                "bg_fanout_min": min(bg_fanout_vector),
                "bg_fanout_max": max(bg_fanout_vector),
                "bg_fanout_sum": sum(bg_fanout_vector),
                "bg_inner_iters": item["background"]["bg_inner_iters"],
                "bg_pre_replays": item["background"]["bg_pre_replays"],
            }
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cp_size",
                "target_mode",
                "target_bs_per_rank",
                "target_bs_min",
                "target_bs_max",
                "target_bs_sum",
                "target_fanout_per_rank",
                "target_fanout_min",
                "target_fanout_max",
                "target_fanout_sum",
                "payload",
                "idle_event_slowest_us",
                "loaded_event_slowest_us",
                "delta_event_slowest_us",
                "slowdown_ratio",
                "idle_preamble_kernel_slowest_us",
                "loaded_preamble_kernel_slowest_us",
                "idle_all2all_kernel_slowest_us",
                "loaded_all2all_kernel_slowest_us",
                "idle_comm_region_kernel_slowest_us",
                "loaded_comm_region_kernel_slowest_us",
                "delta_comm_region_kernel_slowest_us",
                "bg_payload",
                "bg_bs_per_rank",
                "bg_bs_min",
                "bg_bs_max",
                "bg_bs_sum",
                "bg_fanout_per_rank",
                "bg_fanout_min",
                "bg_fanout_max",
                "bg_fanout_sum",
                "bg_inner_iters",
                "bg_pre_replays",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    ensure_dlslime()
    if args.num_requests <= 0:
        raise ValueError("--num-requests must be positive")
    if args.bg_bs_per_rank <= 0:
        raise ValueError("--bg-bs-per-rank must be positive")
    if args.bg_inner_iters <= 0:
        raise ValueError("--bg-inner-iters must be positive")
    if args.bg_pre_replays <= 0:
        raise ValueError("--bg-pre-replays must be positive")

    dtype = get_dtype(args.dtype)
    cp_sizes = parse_cp_sizes(args.cp_sizes)
    kernel_impl = get_kernel_impl(args.dlslime_impl)
    global_rank, world_size, device = init_dist()
    max_cp_size = max(cp_sizes)
    if max_cp_size > world_size:
        raise ValueError(f"Requested max CP size {max_cp_size}, but world_size is only {world_size}")

    summary_path = Path(args.summary_path)
    csv_path = Path(args.csv_path)
    trace_root = Path(args.trace_dir)
    if global_rank == 0:
        trace_root.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    for cp_size in cp_sizes:
        group = make_group(cp_size)
        active = global_rank < cp_size
        if active:
            target_bs_vector = resolve_bs_vector(args.target_bs_per_rank, args.num_requests, cp_size, "--target-bs-per-rank")
            bg_bs_vector = resolve_bs_vector(args.bg_bs_per_rank_list, args.bg_bs_per_rank, cp_size, "--bg-bs-per-rank-list")
            target_fanout_vector = resolve_fanout_vector(args.target_fanout_per_rank, cp_size, "--target-fanout-per-rank")
            bg_fanout_vector = resolve_fanout_vector(args.bg_fanout_per_rank, cp_size, "--bg-fanout-per-rank")
            payload_results = benchmark_cp_size(args, group, global_rank, device, cp_size, dtype, kernel_impl, trace_root)
            if dist.get_rank(group=group) == 0:
                item = {
                    "cp_size": cp_size,
                    "backend": "dlslime_all_to_all_buffer",
                    "dlslime_impl": args.dlslime_impl,
                    "dlslime_file": str(getattr(dlslime, "__file__", "")),
                    "mode": args.mode,
                    "preamble": args.preamble,
                    "dtype": args.dtype,
                    "num_heads": args.num_heads,
                    "head_dim": args.head_dim,
                    "v_head_dim": args.v_head_dim,
                    "batch_size": None if args.target_bs_per_rank.strip() else 1,
                    "num_requests": args.num_requests,
                    "target": {
                        "mode": "remote_all_to_all" if args.target_bs_per_rank.strip() else "single_request_mla",
                        "bs_per_rank": list(target_bs_vector),
                        "fanout_per_rank": list(target_fanout_vector),
                    },
                    "background": {
                        "payload": args.bg_payload,
                        "bs_per_rank": list(bg_bs_vector),
                        "fanout_per_rank": list(bg_fanout_vector),
                        "traffic_pattern": "remote_all_to_all" if args.bg_bs_per_rank_list.strip() else "dense_all_to_all",
                        "feature_dim": background_feature_dim(
                            args.bg_payload, args.num_heads, args.head_dim, args.v_head_dim
                        ),
                        "bg_inner_iters": args.bg_inner_iters if args.mode == "graph" else 1,
                        "bg_pre_replays": args.bg_pre_replays,
                        "bg_preamble": args.bg_preamble,
                    },
                    "payloads": payload_results,
                    "estimated_total_idle_event_slowest_us": estimate_total(payload_results, "idle_event_active_summary"),
                    "estimated_total_loaded_event_slowest_us": estimate_total(
                        payload_results, "loaded_event_active_summary"
                    ),
                    "estimated_total_idle_preamble_slowest_us": estimate_total(
                        payload_results, "idle_preamble_active_summary"
                    ),
                    "estimated_total_loaded_preamble_slowest_us": estimate_total(
                        payload_results, "loaded_preamble_active_summary"
                    ),
                    "estimated_total_idle_all2all_slowest_us": estimate_total(
                        payload_results, "idle_all2all_active_summary"
                    ),
                    "estimated_total_loaded_all2all_slowest_us": estimate_total(
                        payload_results, "loaded_all2all_active_summary"
                    ),
                    "estimated_total_idle_comm_region_slowest_us": estimate_total(
                        payload_results, "idle_comm_region_active_summary"
                    ),
                    "estimated_total_loaded_comm_region_slowest_us": estimate_total(
                        payload_results, "loaded_comm_region_active_summary"
                    ),
                }
                item["estimated_total_delta_event_slowest_us"] = (
                    item["estimated_total_loaded_event_slowest_us"] - item["estimated_total_idle_event_slowest_us"]
                )
                item["estimated_total_delta_comm_region_slowest_us"] = (
                    item["estimated_total_loaded_comm_region_slowest_us"]
                    - item["estimated_total_idle_comm_region_slowest_us"]
                )
                all_results.append(item)
        dist.barrier()

    if global_rank == 0:
        summary = {
            "description": "DLSlime AllToAllBuffer communication estimate across CP sizes",
            "backend": "dlslime_all_to_all_buffer",
            "dlslime_impl": args.dlslime_impl,
            "dlslime_file": str(getattr(dlslime, "__file__", "")),
            "build_intra_ops": bool(getattr(slime_c, "_BUILD_INTRA_OPS", False)),
            "master_rank": MASTER_RANK,
            "cp_sizes": cp_sizes,
            "mode": args.mode,
            "preamble": args.preamble,
            "dtype": args.dtype,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
            "v_head_dim": args.v_head_dim,
            "batch_size": None if args.target_bs_per_rank.strip() else 1,
            "num_requests": args.num_requests,
            "target": {
                "mode": "remote_all_to_all" if args.target_bs_per_rank.strip() else "single_request_mla",
                "bs_per_rank": list(resolve_bs_vector(args.target_bs_per_rank, args.num_requests, max_cp_size, "--target-bs-per-rank")),
                "fanout_per_rank": list(resolve_fanout_vector(args.target_fanout_per_rank, max_cp_size, "--target-fanout-per-rank")),
            },
            "background": {
                "payload": args.bg_payload,
                "bs_per_rank": list(resolve_bs_vector(args.bg_bs_per_rank_list, args.bg_bs_per_rank, max_cp_size, "--bg-bs-per-rank-list")),
                "fanout_per_rank": list(resolve_fanout_vector(args.bg_fanout_per_rank, max_cp_size, "--bg-fanout-per-rank")),
                "traffic_pattern": "remote_all_to_all" if args.bg_bs_per_rank_list.strip() else "dense_all_to_all",
                "feature_dim": background_feature_dim(args.bg_payload, args.num_heads, args.head_dim, args.v_head_dim),
                "bg_inner_iters": args.bg_inner_iters if args.mode == "graph" else 1,
                "bg_pre_replays": args.bg_pre_replays,
                "bg_preamble": args.bg_preamble,
            },
            "results": all_results,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        write_csv(all_results, csv_path)

        print("Saved summary to", summary_path.resolve())
        print("Saved csv to", csv_path.resolve())
        for item in all_results:
            print(
                f"cp_size={item['cp_size']} | "
                f"idle_total_event_slowest_us={item['estimated_total_idle_event_slowest_us']:.2f} | "
                f"loaded_total_event_slowest_us={item['estimated_total_loaded_event_slowest_us']:.2f} | "
                f"delta_total_event_slowest_us={item['estimated_total_delta_event_slowest_us']:.2f}"
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
