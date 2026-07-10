#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import csv
import gc
import json
import math
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanodeploy._cpp import BlockContextSlot, SequenceStatus
from nanodeploy.engine.sequence import Sequence
from nanodeploy.sampling_params import SamplingParams


DATASET_MODES = {"short_only", "natural", "one_long", "long_heavy", "tail_stress"}
LENGTH_SHAPES = {"uniform", "one_long", "two_long"}
SHORT_TOTAL_LEN_THRESHOLD = 8 * 1024
LONG_TOTAL_LEN_THRESHOLD = 64 * 1024
TAIL_TOTAL_LEN_THRESHOLD = 512 * 1024


@dataclass(frozen=True)
class DatasetRequest:
    prompt_len: int
    output_len: int

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len


@dataclass(frozen=True)
class ProfileCase:
    batch_size: int
    context_lens: list[int]
    dop: int
    profile_kind: str
    dataset_name: str = ""
    dataset_mode: str = ""
    sample_seed: int | None = None
    sample_repeat: int | None = None
    decode_step_offset: int = 0
    length_shape: str = "uniform"
    w_attn_bucket: int | None = None
    generation_error: str = ""

    @property
    def w_attn(self) -> int:
        return sum(self.context_lens)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_token_count(value: str) -> int:
    text = value.strip().replace("_", "")
    if not text:
        raise ValueError("empty token count")
    suffix = text[-1].lower()
    multiplier = 1
    if suffix == "k":
        multiplier = 1024
        text = text[:-1]
    elif suffix == "m":
        multiplier = 1024 * 1024
        text = text[:-1]
    return int(math.ceil(float(text) * multiplier))


def parse_token_count_list(value: str) -> list[int]:
    return [parse_token_count(item) for item in value.split(",") if item.strip()]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = math.ceil((pct / 100.0) * len(sorted_values)) - 1
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


def length_stats(context_lens: list[int]) -> dict[str, int | float]:
    if not context_lens:
        return {
            "B": 0,
            "L_avg": 0.0,
            "L_p50": 0,
            "L_p90": 0,
            "L_max": 0,
            "W_attn": 0,
        }
    return {
        "B": len(context_lens),
        "L_avg": statistics.mean(context_lens),
        "L_p50": int(percentile([float(x) for x in context_lens], 50)),
        "L_p90": int(percentile([float(x) for x in context_lens], 90)),
        "L_max": max(context_lens),
        "W_attn": sum(context_lens),
    }


def load_dataset_requests(path: Path) -> list[DatasetRequest]:
    requests: list[DatasetRequest] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        missing = {"prompt_len", "output_len"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            try:
                prompt_len = int(float(row["prompt_len"]))
                output_len = int(float(row["output_len"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: invalid prompt/output length") from exc
            if prompt_len < 1 or output_len < 1:
                continue
            requests.append(DatasetRequest(prompt_len=prompt_len, output_len=output_len))
    if not requests:
        raise ValueError(f"{path} has no usable rows")
    return requests


def _sample_without_replacement(
    rng: random.Random,
    pool: list[DatasetRequest],
    count: int,
    label: str,
) -> list[DatasetRequest]:
    if count <= 0:
        return []
    if len(pool) < count:
        raise ValueError(f"not enough active {label} rows: need {count}, found {len(pool)}")
    return rng.sample(pool, count)


def sample_dataset_batch(
    requests: list[DatasetRequest],
    *,
    mode: str,
    batch_size: int,
    decode_step_offset: int,
    rng: random.Random,
) -> list[int]:
    active = [row for row in requests if decode_step_offset < row.output_len]
    short = [row for row in active if row.total_len < SHORT_TOTAL_LEN_THRESHOLD]
    long = [row for row in active if row.total_len >= LONG_TOTAL_LEN_THRESHOLD]
    tail = [row for row in active if row.total_len >= TAIL_TOTAL_LEN_THRESHOLD]

    if mode == "short_only":
        rows = _sample_without_replacement(rng, short, batch_size, "short_only")
    elif mode == "natural":
        rows = _sample_without_replacement(rng, active, batch_size, "natural")
    elif mode == "one_long":
        rows = _sample_without_replacement(rng, long, 1, "long")
        rows.extend(_sample_without_replacement(rng, short, batch_size - 1, "short"))
        rng.shuffle(rows)
    elif mode == "long_heavy":
        rows = _sample_without_replacement(rng, long, batch_size, "long_heavy")
    elif mode == "tail_stress":
        rows = _sample_without_replacement(rng, tail, batch_size, "tail_stress")
    else:
        raise ValueError(f"unsupported dataset mode: {mode}")

    return [row.prompt_len + decode_step_offset for row in rows]


def context_lens_for_w_attn(
    *,
    batch_size: int,
    w_attn: int,
    shape: str,
    short_context_len: int,
) -> list[int]:
    if shape == "uniform":
        return [math.ceil(w_attn / batch_size)] * batch_size

    long_count = {"one_long": 1, "two_long": 2}.get(shape)
    if long_count is None:
        raise ValueError(f"unsupported length shape: {shape}")
    if batch_size < long_count:
        raise ValueError(f"shape {shape} requires B >= {long_count}")

    short_count = batch_size - long_count
    short_total = short_count * short_context_len
    long_total = w_attn - short_total
    if long_total < long_count:
        raise ValueError(
            f"W_attn={w_attn} is too small for {shape} with "
            f"{short_count} short requests of length {short_context_len}"
        )

    base = long_total // long_count
    extra = long_total % long_count
    long_lens = [base + (1 if idx < extra else 0) for idx in range(long_count)]
    return long_lens + [short_context_len] * short_count


def parse_name_list(value: str, *, valid: set[str], flag_name: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in names if item not in valid]
    if invalid:
        raise ValueError(f"{flag_name} has unsupported values: {invalid}")
    return names


def same_local_group(ranks: list[int], local_group_size: int) -> bool:
    if not ranks:
        return True
    groups = {rank // local_group_size for rank in ranks}
    return len(groups) == 1


def infer_node_local(
    *,
    occupied: list[int],
    append: list[int],
    sp_send_counts: list[int],
    sp_recv_counts: list[int],
    local_group_size: int,
) -> bool:
    active = sorted(set(occupied) | set(append))
    if not same_local_group(active, local_group_size):
        return False
    active_set = set(active)
    for rank, count in enumerate(sp_send_counts):
        if count and rank not in active_set:
            return False
    for rank, count in enumerate(sp_recv_counts):
        if count and rank not in active_set:
            return False
    return True


def row_base(
    args: argparse.Namespace,
    case: ProfileCase,
    *,
    viable: bool,
    skip_reason: str = "",
    node_local: bool | None = None,
) -> dict[str, Any]:
    stats = length_stats(case.context_lens)
    uniform_len = case.context_lens[0] if case.context_lens and len(set(case.context_lens)) == 1 else ""
    return {
        "profile_kind": case.profile_kind,
        "dataset_name": case.dataset_name,
        "dataset_mode": case.dataset_mode,
        "sample_seed": case.sample_seed if case.sample_seed is not None else "",
        "sample_repeat": case.sample_repeat if case.sample_repeat is not None else "",
        "decode_step_offset": case.decode_step_offset,
        "length_shape": case.length_shape,
        "W_attn_bucket": case.w_attn_bucket if case.w_attn_bucket is not None else "",
        "B": case.batch_size,
        "L_avg": stats["L_avg"],
        "L_p50": stats["L_p50"],
        "L_p90": stats["L_p90"],
        "L_max": stats["L_max"],
        "W_attn": stats["W_attn"],
        "d_attn": case.dop,
        "viable": viable,
        "skip_reason": skip_reason,
        "node_local": "" if node_local is None else node_local,
        "batch_size": case.batch_size,
        "context_len": uniform_len,
        "context_lens": case.context_lens,
        "w_attn": stats["W_attn"],
        "dop": case.dop,
        "sp": args.sp,
        "ep": args.ep,
        "warmup": args.warmup,
        "steps": args.steps,
        "min_comp_bound_batch_size": max(1, math.ceil(case.batch_size / case.dop)),
    }


def validate_case(args: argparse.Namespace, case: ProfileCase) -> str:
    if case.generation_error:
        return case.generation_error
    if case.batch_size != len(case.context_lens):
        return f"B={case.batch_size} but generated {len(case.context_lens)} lengths"
    if case.dop < 1 or case.dop > args.sp:
        return f"d={case.dop} is outside [1, SP={args.sp}]"
    if any(length < case.dop for length in case.context_lens):
        return f"L_i must be >= d={case.dop}"
    if max(case.context_lens, default=0) > args.max_model_len:
        return (
            f"L_max={max(case.context_lens)} exceeds "
            f"--max-model-len={args.max_model_len}"
        )
    return ""


def split_tokens(total: int, ranks: list[int], sp: int) -> list[int]:
    tokens = [0] * sp
    base = total // len(ranks)
    extra = total % len(ranks)
    for idx, rank in enumerate(ranks):
        tokens[rank] = base + (1 if idx < extra else 0)
    return tokens


def make_decode_sequence(
    *,
    seq_idx: int,
    context_len: int,
    sp: int,
    dop: int,
    engine_id: str,
) -> Sequence:
    seq = Sequence(
        [1000 + seq_idx],
        sampling_params=SamplingParams(max_tokens=1, ignore_eos=True),
        engine_id=engine_id,
        master_sp_rank=seq_idx % dop,
    )
    seq.active(engine_id, sp, 1)
    seq.num_tokens = context_len
    seq.num_prompt_tokens = context_len
    seq.num_checkpointed_tokens = context_len
    seq.last_token = 1000 + seq_idx
    seq.status = SequenceStatus.RUNNING

    participants = list(range(dop))
    master = seq_idx % dop
    ctx = seq.block_ctx(BlockContextSlot.ACTIVE)
    ctx.dp_idx = 0
    ctx.master_sp_idx = master
    ctx.append_sp_idx = -1
    ctx.num_dispatched_tokens = split_tokens(context_len, participants, sp)
    if ctx.num_dispatched_tokens[master] <= 0:
        raise ValueError(
            f"context_len={context_len} is too small for dop={dop}; "
            f"master rank {master} has no KV tokens"
        )
    return seq


def enqueue_batch(
    llm: LLM,
    *,
    context_lens: list[int],
    sp: int,
    dop: int,
) -> list[Sequence]:
    worker = llm.scheduler.worker_state[0]
    seqs = [
        make_decode_sequence(
            seq_idx=idx,
            context_len=context_len,
            sp=sp,
            dop=dop,
            engine_id=llm.engine_id,
        )
        for idx, context_len in enumerate(context_lens)
    ]
    for seq in seqs:
        seq.metric = llm.metrics_manager.create_sequence_metric(seq.seq_id, seq.num_prompt_tokens)
        worker.allocate(seq)
        worker.running.append(seq)
    return seqs


def run_decode_step_with_schedule(llm: LLM) -> dict[str, Any]:
    tp_size = llm.config.attention_tp

    step_start = time.perf_counter()
    sch_begin = time.perf_counter()
    sch_res = llm.scheduler.schedule()
    sch_ms = (time.perf_counter() - sch_begin) * 1000.0

    dp_sp_tp_seqs = [seqs for seqs in sch_res.dp_sp_seqs for _ in range(tp_size)]

    model_begin = time.perf_counter()
    token_ids = llm.executor.run(dp_sp_tp_seqs, sch_res.is_prefill)[::tp_size]
    model_ms = (time.perf_counter() - model_begin) * 1000.0

    post_begin = time.perf_counter()
    llm.scheduler.postprocess(
        sch_res.filtered_dp_sp_seqs,
        token_ids,
        None,
        (time.perf_counter() - step_start) * 1000.0,
        llm.config.loop_count,
    )
    post_ms = (time.perf_counter() - post_begin) * 1000.0
    step_ms = (time.perf_counter() - step_start) * 1000.0

    real_seqs = [
        seq
        for seq in sch_res.dp_seqs[0]
        if seq not in llm.scheduler.worker_state[0].dummy_seqs
    ]
    master_counts = Counter(
        seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx for seq in real_seqs
    )
    sp_size_hist = {}
    if sch_res.sp_size_hist_per_dp:
        sp_size_hist = {
            idx: count
            for idx, count in enumerate(sch_res.sp_size_hist_per_dp[0])
            if count
        }

    return {
        "step_ms": step_ms,
        "model_ms": model_ms,
        "scheduler_ms": sch_ms,
        "postprocess_ms": post_ms,
        "batch_size_scheduled": len(real_seqs),
        "master_counts": dict(sorted(master_counts.items())),
        "occupied": list(sch_res.loongserve_occupied_instances[0])
        if sch_res.loongserve_occupied_instances
        else [],
        "append": list(sch_res.loongserve_append_instances[0])
        if sch_res.loongserve_append_instances
        else [],
        "draining": list(sch_res.loongserve_draining_instances[0])
        if sch_res.loongserve_draining_instances
        else [],
        "sp_send_counts": list(sch_res.sp_send_counts[0])
        if sch_res.sp_send_counts
        else [],
        "sp_recv_counts": list(sch_res.sp_recv_counts[0])
        if sch_res.sp_recv_counts
        else [],
        "sp_size_hist": sp_size_hist,
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, float]:
    step_values = [sample["step_ms"] for sample in samples]
    model_values = [sample["model_ms"] for sample in samples]
    sch_values = [sample["scheduler_ms"] for sample in samples]
    post_values = [sample["postprocess_ms"] for sample in samples]
    return {
        "step_mean_ms": statistics.mean(step_values),
        "step_p50_ms": statistics.median(step_values),
        "step_p90_ms": percentile(step_values, 90),
        "model_mean_ms": statistics.mean(model_values),
        "model_p50_ms": statistics.median(model_values),
        "model_p90_ms": percentile(model_values, 90),
        "scheduler_mean_ms": statistics.mean(sch_values),
        "postprocess_mean_ms": statistics.mean(post_values),
    }


def build_llm(args: argparse.Namespace, *, batch_size: int, dop: int) -> LLM:
    from nanodeploy import LLM

    min_comp_bound = max(1, math.ceil(batch_size / dop))
    return LLM(
        args.model_path,
        enforce_eager=False,
        cuda_graph_mode="full",
        attention_dp=1,
        attention_sp=args.sp,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=args.ep,
        ffn_tp=1,
        mode="decode",
        master_address=args.master_address,
        ray_address=args.ray_address,
        dummy_prefill=True,
        dummy_weight=True,
        perfect_eplb=True,
        max_num_seqs=batch_size,
        max_num_recv_seqs=batch_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max(args.max_model_len, batch_size * 4),
        loop_count=1,
        kvcache_block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        scheduler_mode="centralized",
        sp_backend="nccl",
        loongserve_decode_scheduler=True,
        loongserve_min_comp_bound_batch_size=min_comp_bound,
        loongserve_max_local_decode_sp=args.local_group_size,
    )


def profile_point(
    llm: LLM,
    args: argparse.Namespace,
    *,
    case: ProfileCase,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    warmup_samples: list[dict[str, Any]] = []
    total_steps = args.warmup + args.steps
    for step_idx in range(total_steps):
        enqueue_batch(
            llm,
            context_lens=case.context_lens,
            sp=args.sp,
            dop=case.dop,
        )
        sample = run_decode_step_with_schedule(llm)
        if sample["batch_size_scheduled"] != case.batch_size:
            raise RuntimeError(
                f"scheduled batch mismatch: expected {case.batch_size}, "
                f"got {sample['batch_size_scheduled']}"
            )
        if step_idx < args.warmup:
            warmup_samples.append(sample)
        else:
            samples.append(sample)

    summary = summarize(samples)
    first = samples[0]
    node_local = infer_node_local(
        occupied=first["occupied"],
        append=first["append"],
        sp_send_counts=first["sp_send_counts"],
        sp_recv_counts=first["sp_recv_counts"],
        local_group_size=args.local_group_size,
    )
    return {
        **row_base(args, case, viable=True, node_local=node_local),
        "batch_size_scheduled": first["batch_size_scheduled"],
        "master_counts": first["master_counts"],
        "occupied": first["occupied"],
        "append": first["append"],
        "draining": first["draining"],
        "sp_size_hist": first["sp_size_hist"],
        "sp_send_counts": first["sp_send_counts"],
        "sp_recv_counts": first["sp_recv_counts"],
        "warmup_step_mean_ms": statistics.mean(
            [sample["step_ms"] for sample in warmup_samples]
        )
        if warmup_samples
        else 0.0,
        **summary,
    }


def write_results(out_path: Path, rows: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    csv_path = out_path.with_suffix(".csv")
    fieldnames = [
        "profile_kind",
        "dataset_name",
        "dataset_mode",
        "sample_seed",
        "sample_repeat",
        "decode_step_offset",
        "length_shape",
        "W_attn_bucket",
        "B",
        "L_avg",
        "L_p50",
        "L_p90",
        "L_max",
        "W_attn",
        "d_attn",
        "viable",
        "skip_reason",
        "node_local",
        "batch_size",
        "context_len",
        "context_lens",
        "w_attn",
        "dop",
        "sp",
        "ep",
        "warmup",
        "steps",
        "batch_size_scheduled",
        "step_mean_ms",
        "step_p50_ms",
        "step_p90_ms",
        "model_mean_ms",
        "model_p50_ms",
        "model_p90_ms",
        "scheduler_mean_ms",
        "postprocess_mean_ms",
        "warmup_step_mean_ms",
        "min_comp_bound_batch_size",
        "master_counts",
        "occupied",
        "append",
        "draining",
        "sp_size_hist",
        "sp_send_counts",
        "sp_recv_counts",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def print_table(rows: list[dict[str, Any]]) -> None:
    print(
        "profile_kind,B,L_avg,L_p90,L_max,d,W_attn,viable,node_local,skip_reason,step_mean_ms,step_p50_ms,step_p90_ms,model_mean_ms,scheduler_mean_ms,postprocess_mean_ms",
        flush=True,
    )
    for row in rows:
        if not row.get("viable", False):
            print(
                f"{row.get('profile_kind','')},{row.get('B','')},"
                f"{row.get('L_avg','')},{row.get('L_p90','')},"
                f"{row.get('L_max','')},{row.get('d_attn','')},"
                f"{row.get('W_attn','')},False,{row.get('node_local','')},"
                f"{row.get('skip_reason','')},,,,,,",
                flush=True,
            )
            continue
        print(
            f"{row['profile_kind']},{row['B']},{row['L_avg']:.2f},"
            f"{row['L_p90']},{row['L_max']},{row['d_attn']},"
            f"{row['W_attn']},{row['viable']},{row['node_local']},,"
            f"{row['step_mean_ms']:.4f},"
            f"{row['step_p50_ms']:.4f},{row['step_p90_ms']:.4f},"
            f"{row['model_mean_ms']:.4f},{row['scheduler_mean_ms']:.4f},"
            f"{row['postprocess_mean_ms']:.4f}",
            flush=True,
        )


def build_cases_for_batch_dop(
    args: argparse.Namespace,
    *,
    batch_size: int,
    dop: int,
    dataset_requests: list[DatasetRequest] | None,
    length_shapes: list[str],
    lengths: list[int],
    w_attn_buckets: list[int],
    decode_step_offsets: list[int],
) -> list[ProfileCase]:
    cases: list[ProfileCase] = []
    if dataset_requests is not None:
        dataset_path = Path(args.dataset_path)
        dataset_name = dataset_path.stem
        for offset in decode_step_offsets:
            for repeat_idx in range(args.sample_repeat):
                case_seed = args.sample_seed + repeat_idx
                try:
                    context_lens = sample_dataset_batch(
                        dataset_requests,
                        mode=args.dataset_mode,
                        batch_size=batch_size,
                        decode_step_offset=offset,
                        rng=random.Random(case_seed),
                    )
                    cases.append(
                        ProfileCase(
                            batch_size=batch_size,
                            context_lens=context_lens,
                            dop=dop,
                            profile_kind="dataset_replay",
                            dataset_name=dataset_name,
                            dataset_mode=args.dataset_mode,
                            sample_seed=case_seed,
                            sample_repeat=repeat_idx,
                            decode_step_offset=offset,
                        )
                    )
                except ValueError as exc:
                    cases.append(
                        ProfileCase(
                            batch_size=batch_size,
                            context_lens=[],
                            dop=dop,
                            profile_kind="dataset_replay",
                            dataset_name=dataset_name,
                            dataset_mode=args.dataset_mode,
                            sample_seed=case_seed,
                            sample_repeat=repeat_idx,
                            decode_step_offset=offset,
                            generation_error=str(exc),
                        )
                    )
        return cases

    if w_attn_buckets:
        for w_attn in w_attn_buckets:
            for shape in length_shapes:
                try:
                    context_lens = context_lens_for_w_attn(
                        batch_size=batch_size,
                        w_attn=w_attn,
                        shape=shape,
                        short_context_len=args.short_context_len,
                    )
                    cases.append(
                        ProfileCase(
                            batch_size=batch_size,
                            context_lens=context_lens,
                            dop=dop,
                            profile_kind="uniform",
                            length_shape=shape,
                            w_attn_bucket=w_attn,
                        )
                    )
                except ValueError as exc:
                    case = ProfileCase(
                        batch_size=batch_size,
                        context_lens=[],
                        dop=dop,
                        profile_kind="uniform",
                        length_shape=shape,
                        w_attn_bucket=w_attn,
                        generation_error=str(exc),
                    )
                    cases.append(case)
        return cases

    for length in lengths:
        total_w_attn = batch_size * length
        for shape in length_shapes:
            try:
                if shape == "uniform":
                    context_lens = [length] * batch_size
                else:
                    context_lens = context_lens_for_w_attn(
                        batch_size=batch_size,
                        w_attn=total_w_attn,
                        shape=shape,
                        short_context_len=args.short_context_len,
                    )
                cases.append(
                    ProfileCase(
                        batch_size=batch_size,
                        context_lens=context_lens,
                        dop=dop,
                        profile_kind="uniform",
                        length_shape=shape,
                        w_attn_bucket=total_w_attn if shape != "uniform" else None,
                    )
                )
            except ValueError as exc:
                case = ProfileCase(
                    batch_size=batch_size,
                    context_lens=[],
                    dop=dop,
                    profile_kind="uniform",
                    length_shape=shape,
                    w_attn_bucket=total_w_attn,
                    generation_error=str(exc),
                )
                cases.append(case)
    return cases


def compact_error(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    return message or type(exc).__name__


def teardown_llm(llm: Any, teardown_sleep: float) -> None:
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()
    gc.collect()
    time.sleep(teardown_sleep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile LoongServe-style decode DoP on a fixed SP/EP cluster."
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/nvme1n1/ml_research/chenjiefei/models/deepseek-v3",
    )
    parser.add_argument("--ray-address", default="10.102.252.174:6380")
    parser.add_argument("--master-address", default="10.102.252.174:29644")
    parser.add_argument("--sp", type=int, default=16)
    parser.add_argument("--ep", type=int, default=16)
    parser.add_argument("--batches", default="16")
    parser.add_argument("--lengths", default="1024,4096,16384")
    parser.add_argument("--w-attn-buckets", default="")
    parser.add_argument("--length-shapes", default="uniform")
    parser.add_argument("--short-context-len", type=int, default=1024)
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--dataset-mode", default="natural", choices=sorted(DATASET_MODES))
    parser.add_argument("--decode-step-offsets", default="0")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--sample-repeat", type=int, default=1)
    parser.add_argument("--dops", default="1,2,4,8")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--local-group-size", type=int, default=8)
    parser.add_argument("--teardown-sleep", type=float, default=3.0)
    parser.add_argument(
        "--out",
        default="docs-dev/profile-results/loongserve_16g_pilot.jsonl",
    )
    args = parser.parse_args()

    batches = parse_int_list(args.batches)
    lengths = parse_token_count_list(args.lengths)
    w_attn_buckets = parse_token_count_list(args.w_attn_buckets)
    length_shapes = parse_name_list(
        args.length_shapes,
        valid=LENGTH_SHAPES,
        flag_name="--length-shapes",
    )
    decode_step_offsets = parse_int_list(args.decode_step_offsets)
    dops = parse_int_list(args.dops)
    if any(dop < 1 or dop > args.sp for dop in dops):
        raise ValueError("all dops must be in [1, sp]")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.sample_repeat < 1:
        raise ValueError("--sample-repeat must be >= 1")
    if args.short_context_len < 1:
        raise ValueError("--short-context-len must be >= 1")
    if args.local_group_size < 1:
        raise ValueError("--local-group-size must be >= 1")
    if args.dataset_path and w_attn_buckets:
        raise ValueError("--dataset-path and --w-attn-buckets are mutually exclusive")

    dataset_requests = (
        load_dataset_requests(Path(args.dataset_path)) if args.dataset_path else None
    )

    rows: list[dict[str, Any]] = []
    out_path = Path(args.out)

    for batch_size in batches:
        for dop in dops:
            cases = build_cases_for_batch_dop(
                args,
                batch_size=batch_size,
                dop=dop,
                dataset_requests=dataset_requests,
                length_shapes=length_shapes,
                lengths=lengths,
                w_attn_buckets=w_attn_buckets,
                decode_step_offsets=decode_step_offsets,
            )
            runnable_cases: list[ProfileCase] = []
            for case in cases:
                skip_reason = validate_case(args, case)
                if skip_reason:
                    rows.append(row_base(args, case, viable=False, skip_reason=skip_reason))
                    write_results(out_path, rows)
                    print(
                        f"skip B={case.batch_size}, d={case.dop}, "
                        f"W={case.w_attn or case.w_attn_bucket}: {skip_reason}",
                        flush=True,
                    )
                else:
                    runnable_cases.append(case)
            if not runnable_cases:
                continue

            print(
                f"=== init LLM for B={batch_size}, d={dop}, SP={args.sp}, EP={args.ep} ===",
                flush=True,
            )
            try:
                llm = build_llm(args, batch_size=batch_size, dop=dop)
            except Exception as exc:
                reason = f"llm_init_failed: {compact_error(exc)}"
                for case in runnable_cases:
                    rows.append(row_base(args, case, viable=False, skip_reason=reason))
                write_results(out_path, rows)
                print(f"skip B={batch_size}, d={dop}: {reason}", flush=True)
                continue

            for case_idx, case in enumerate(runnable_cases):
                stats = length_stats(case.context_lens)
                print(
                    f"--- profile {case.profile_kind} B={batch_size}, "
                    f"L_avg={stats['L_avg']:.2f}, L_p90={stats['L_p90']}, "
                    f"L_max={stats['L_max']}, W={stats['W_attn']}, d={dop} ---",
                    flush=True,
                )
                try:
                    row = profile_point(llm, args, case=case)
                    rows.append(row)
                    print(
                        f"RESULT B={batch_size} W={row['W_attn']} d={dop}: "
                        f"step_mean={row['step_mean_ms']:.4f}ms "
                        f"model_mean={row['model_mean_ms']:.4f}ms "
                        f"node_local={row['node_local']}",
                        flush=True,
                    )
                except Exception as exc:
                    reason = f"profile_failed: {compact_error(exc)}"
                    rows.append(row_base(args, case, viable=False, skip_reason=reason))
                    print(f"skip B={batch_size}, d={dop}: {reason}", flush=True)
                    teardown_llm(llm, args.teardown_sleep)
                    if case_idx == len(runnable_cases) - 1:
                        llm = None
                        write_results(out_path, rows)
                        break
                    try:
                        llm = build_llm(args, batch_size=batch_size, dop=dop)
                    except Exception as reinit_exc:
                        reinit_reason = f"llm_reinit_failed: {compact_error(reinit_exc)}"
                        for remaining in runnable_cases[case_idx + 1 :]:
                            rows.append(
                                row_base(
                                    args,
                                    remaining,
                                    viable=False,
                                    skip_reason=reinit_reason,
                                )
                            )
                        print(f"skip remaining B={batch_size}, d={dop}: {reinit_reason}", flush=True)
                        llm = None
                        write_results(out_path, rows)
                        break
                write_results(out_path, rows)

            if llm is not None:
                teardown_llm(llm, args.teardown_sleep)

    write_results(out_path, rows)
    print_table(rows)
    print(f"JSONL: {out_path}", flush=True)
    print(f"CSV: {out_path.with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    main()
