#!/usr/bin/env python3
"""Validate and compare interleaved centralized/hierarchical qdiag runs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


STAGE_FIELDS = (
    "ingress_drain_ms",
    "schedule_ms",
    "consensus_exposed_wait_ms",
    "consensus_overlap_window_ms",
    "leader_arrival_skew_ms",
    "leader_rendezvous_ms",
    "late_participant_collective_ms",
    "execute_ms",
    "postprocess_ms",
    "quantum_total_ms",
    "legacy_consensus_wait_ms",
    "legacy_consensus_total_ms",
)
QUANTUM_CRITICAL_FIELDS = (
    "ingress_drain_ms",
    "leader_arrival_skew_ms",
    "late_participant_collective_ms",
)
EXECUTOR_FIELDS = (
    "actor_submit_latency_ms",
    "worker_command_submit_latency_ms",
    "send_seqs_latency_ms",
    "ray_get_latency_ms",
    "worker_result_wait_latency_ms",
    "worker_finish_to_ray_get_ms",
    "worker_finish_to_result_ms",
    "result_rebuild_ms",
    "result_unpack_ms",
)
WORKER_FIELDS = (
    "recv_seqs_ms",
    "prepare_update_host_ms",
    "forward_host_ms",
    "gpu_loop_ms",
    "loop_host_ms",
    "token_materialize_ms",
    "worker_body_ms",
    "worker_total_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the manifest produced by "
            "run_2node_rate40_qdiag_ab.sh"
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-html", type=Path)
    return parser.parse_args()


def _resolve_artifact(path_value: str, *, relative_to: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL record {path}:{line_number}: {exc}"
                ) from exc
    return records


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: Iterable[float]) -> dict:
    samples = [float(value) for value in values]
    if not samples:
        return {
            "samples": 0,
            "total": 0.0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    return {
        "samples": len(samples),
        "total": sum(samples),
        "mean": statistics.fmean(samples),
        "p50": _percentile(samples, 50),
        "p95": _percentile(samples, 95),
        "p99": _percentile(samples, 99),
    }


def _request_totals(path: Path) -> dict[str, int]:
    prompt_tokens = 0
    output_tokens = 0
    successful_requests = 0
    for record in _read_jsonl(path):
        if record.get("status") != "FINISHED":
            continue
        successful_requests += 1
        prompt = int(record["prompt_tokens"])
        actual_output = int(record["actual_output_tokens"])
        prompt_tokens += prompt
        output_tokens += actual_output
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "successful_requests": successful_requests,
    }


def _critical_worker(record: dict) -> dict | None:
    timings = record.get("executor", {}).get(
        "worker_rank_timings", ()
    )
    if not timings:
        return None
    return max(
        timings,
        key=lambda timing: float(timing["worker_total_ms"]),
    )


def _quantum_summary(records: list[dict], output_tokens: int) -> dict:
    if not records:
        raise ValueError("quantum diagnostic JSONL is empty")
    if output_tokens <= 0:
        raise ValueError("successful output token total must be positive")

    stage_values: dict[str, list[float]] = defaultdict(list)
    executor_values: dict[str, list[float]] = defaultdict(list)
    worker_values: dict[str, list[float]] = defaultdict(list)
    critical_cpu_residuals = []
    useful_batch_sizes = []
    attention_work = []
    gpu_rank_time_ms = 0.0
    worker_cpu_rank_time_ms = 0.0
    rank_attention_work: dict[int, int] = defaultdict(int)
    records_by_quantum: dict[tuple[int, int], list[dict]] = defaultdict(list)

    for record in records:
        records_by_quantum[
            (
                int(record.get("wave_id", 0)),
                int(record["quantum_id"]),
            )
        ].append(record)
        for field in STAGE_FIELDS:
            if field in QUANTUM_CRITICAL_FIELDS:
                continue
            value = record.get(field)
            if value is not None:
                stage_values[field].append(float(value))
        if int(record.get("schema_version", 0)) < 3:
            legacy_wait = record.get("consensus_wait_ms")
            if legacy_wait is not None:
                stage_values["legacy_consensus_wait_ms"].append(
                    float(legacy_wait)
                )
            legacy_total = record.get("consensus_total_ms")
            if legacy_total is not None:
                stage_values["legacy_consensus_total_ms"].append(
                    float(legacy_total)
                )
        executor = record.get("executor", {})
        for field in EXECUTOR_FIELDS:
            value = executor.get(field)
            if value is not None:
                executor_values[field].append(float(value))
        useful_batch_sizes.append(
            float(record.get("useful_real_batch_size", 0))
        )
        attention_work.append(
            float(record.get("attention_work_tokens", 0))
        )

        critical = _critical_worker(record)
        if critical is not None:
            critical_gpu = critical.get("gpu_loop_ms")
            if critical_gpu is not None:
                critical_cpu_residuals.append(
                    max(
                        0.0,
                        float(critical["worker_total_ms"])
                        - float(critical_gpu),
                    )
                )

        for timing in executor.get("worker_rank_timings", ()):
            for field in WORKER_FIELDS:
                value = timing.get(field)
                if value is not None:
                    worker_values[field].append(float(value))
            gpu_ms = timing.get("gpu_loop_ms")
            worker_ms = timing.get("worker_total_ms")
            if gpu_ms is not None:
                gpu_rank_time_ms += float(gpu_ms)
            if gpu_ms is not None and worker_ms is not None:
                worker_cpu_rank_time_ms += max(
                    0.0, float(worker_ms) - float(gpu_ms)
                )

        for rank_load in record.get("rank_loads_before", ()):
            rank_attention_work[int(rank_load["global_rank"])] += int(
                rank_load.get("active_dispatched_tokens", 0)
            )

    for quantum_records in records_by_quantum.values():
        ingress_values = [
            float(record["ingress_drain_ms"])
            for record in quantum_records
            if record.get("ingress_drain_ms") is not None
        ]
        if ingress_values:
            stage_values["ingress_drain_ms"].append(max(ingress_values))
        arrival_skews = [
            float(record["leader_arrival_skew_ms"])
            for record in quantum_records
            if record.get("leader_arrival_skew_ms") is not None
        ]
        if arrival_skews:
            stage_values["leader_arrival_skew_ms"].append(
                max(arrival_skews)
            )
        late_collective = [
            float(record["late_participant_collective_ms"])
            for record in quantum_records
            if record.get("late_participant_collective_ms") is not None
        ]
        if late_collective:
            stage_values["late_participant_collective_ms"].append(
                max(late_collective)
            )

    rank_totals = list(rank_attention_work.values())
    rank_mean = statistics.fmean(rank_totals) if rank_totals else 0.0
    rank_cv = (
        statistics.pstdev(rank_totals) / rank_mean
        if len(rank_totals) > 1 and rank_mean
        else 0.0
    )
    return {
        "samples": len(records),
        "stage_ms": {
            field: _stats(stage_values[field]) for field in STAGE_FIELDS
        },
        "executor_ms": {
            field: _stats(executor_values[field])
            for field in EXECUTOR_FIELDS
        },
        "worker_ms": {
            field: _stats(worker_values[field]) for field in WORKER_FIELDS
        },
        "critical_worker_cpu_residual_ms": _stats(
            critical_cpu_residuals
        ),
        "useful_real_batch_size": _stats(useful_batch_sizes),
        "attention_work_tokens": _stats(attention_work),
        "gpu_rank_time_ms_total": gpu_rank_time_ms,
        "gpu_rank_time_ms_per_output_token": (
            gpu_rank_time_ms / output_tokens
        ),
        "worker_cpu_rank_time_ms_total": worker_cpu_rank_time_ms,
        "worker_cpu_rank_time_ms_per_output_token": (
            worker_cpu_rank_time_ms / output_tokens
        ),
        "rank_attention_work": {
            "per_global_rank": {
                str(rank): value
                for rank, value in sorted(rank_attention_work.items())
            },
            "coefficient_of_variation": rank_cv,
            "min": min(rank_totals) if rank_totals else 0,
            "max": max(rank_totals) if rank_totals else 0,
        },
    }


def _relative_delta(hierarchical: float, central: float) -> float | None:
    if central == 0:
        return None
    return round((hierarchical / central - 1.0) * 100.0, 6)


def _load_run(row: dict, manifest_dir: Path) -> dict:
    if row["status"] != "success":
        raise ValueError(
            f"manifest run {row['run_index']} is not successful: "
            f"{row['status']}"
        )
    summary_path = _resolve_artifact(
        row["summary_json"], relative_to=manifest_dir
    )
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    request_path = _resolve_artifact(
        summary["request_metrics_jsonl"],
        relative_to=summary_path.parent,
    )
    quantum_path_value = summary.get("quantum_diagnostics_jsonl") or (
        summary.get("hierarchical_quantum_diagnostics_jsonl")
    )
    if not quantum_path_value:
        raise ValueError(f"summary has no quantum JSONL: {summary_path}")
    quantum_path = _resolve_artifact(
        quantum_path_value, relative_to=summary_path.parent
    )
    workload = _request_totals(request_path)
    if workload["successful_requests"] != int(
        summary["successful_requests"]
    ):
        raise ValueError(
            f"request count mismatch in {row['run_index']}: "
            f"summary={summary['successful_requests']} "
            f"jsonl={workload['successful_requests']}"
        )
    if int(summary.get("failed_requests", 0)) != 0:
        raise ValueError(f"run contains failed requests: {row['run_index']}")
    quantum_records = _read_jsonl(quantum_path)
    expected_samples = int(
        summary.get(
            "quantum_diagnostic_samples",
            summary.get("hierarchical_quantum_diagnostic_samples", 0),
        )
    )
    if expected_samples != len(quantum_records):
        raise ValueError(
            f"quantum sample mismatch in {row['run_index']}: "
            f"summary={expected_samples} jsonl={len(quantum_records)}"
        )
    corrected_tpot = float(summary["tpot_with_queue_ms"]["mean"])
    runtime_s = float(summary["benchmark_runtime_s"])
    return {
        "run_id": row["run_index"],
        "stage": row["stage"],
        "scheduler_arch": row["scheduler_arch"],
        "summary_json": str(summary_path),
        "request_metrics_jsonl": str(request_path),
        "quantum_diagnostics_jsonl": str(quantum_path),
        "benchmark_runtime_s": runtime_s,
        "corrected_tpot_mean_ms": corrected_tpot,
        "output_throughput_tokens_s": (
            workload["output_tokens"] / runtime_s
        ),
        "workload": workload,
        "qdiag": _quantum_summary(
            quantum_records, workload["output_tokens"]
        ),
    }


def _build_pairs(runs: list[dict]) -> list[dict]:
    if len(runs) % 2:
        raise ValueError("interleaved manifest must contain an even run count")
    pairs = []
    for pair_index in range(0, len(runs), 2):
        candidates = runs[pair_index : pair_index + 2]
        by_stage = {run["stage"]: run for run in candidates}
        if set(by_stage) != {"central", "hierarchical"}:
            raise ValueError(
                "each adjacent run pair must contain one central and one "
                f"hierarchical run: {[run['run_id'] for run in candidates]}"
            )
        central = by_stage["central"]
        hierarchical = by_stage["hierarchical"]
        central_qdiag = central["qdiag"]
        hierarchical_qdiag = hierarchical["qdiag"]
        pairs.append(
            {
                "pair_index": pair_index // 2 + 1,
                "run_order": [run["stage"] for run in candidates],
                "central_run_id": central["run_id"],
                "hierarchical_run_id": hierarchical["run_id"],
                "corrected_tpot_delta_percent_hierarchical_vs_central": (
                    _relative_delta(
                        hierarchical["corrected_tpot_mean_ms"],
                        central["corrected_tpot_mean_ms"],
                    )
                ),
                "output_throughput_delta_percent_hierarchical_vs_central": (
                    _relative_delta(
                        hierarchical["output_throughput_tokens_s"],
                        central["output_throughput_tokens_s"],
                    )
                ),
                "gpu_rank_time_per_token_delta_percent": _relative_delta(
                    hierarchical_qdiag[
                        "gpu_rank_time_ms_per_output_token"
                    ],
                    central_qdiag[
                        "gpu_rank_time_ms_per_output_token"
                    ],
                ),
                "worker_cpu_rank_time_per_token_delta_percent": (
                    _relative_delta(
                        hierarchical_qdiag[
                            "worker_cpu_rank_time_ms_per_output_token"
                        ],
                        central_qdiag[
                            "worker_cpu_rank_time_ms_per_output_token"
                        ],
                    )
                ),
            }
        )
    return pairs


def _mean_pair_metric(pairs: list[dict], field: str) -> float | None:
    values = [pair[field] for pair in pairs if pair[field] is not None]
    return statistics.fmean(values) if values else None


def _diagnosis(pairs: list[dict]) -> str:
    gpu_delta = _mean_pair_metric(
        pairs, "gpu_rank_time_per_token_delta_percent"
    )
    cpu_delta = _mean_pair_metric(
        pairs, "worker_cpu_rank_time_per_token_delta_percent"
    )
    if gpu_delta is not None and gpu_delta > 1.0:
        return (
            "Hierarchical GPU-rank time per output token is higher by more "
            "than 1%; inspect matched batch/context shapes and packing first."
        )
    if cpu_delta is not None and cpu_delta > 1.0:
        return (
            "GPU-rank time is aligned, but worker CPU residual is higher; "
            "use the worker stage table to identify prepare or materialize."
        )
    return (
        "GPU-rank and worker CPU time per token are within 1% on average; "
        "remaining movement is likely driver/control stages or run variance."
    )


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _metric_mean(qdiag: dict, section: str, *fields: str) -> float | None:
    metrics = qdiag[section]
    for field in fields:
        value = metrics[field]["mean"]
        if value is not None:
            return value
    return None


def _render_html(comparison: dict) -> str:
    runs = comparison["runs"]
    pairs = comparison["pairs"]
    run_rows = []
    for run in runs:
        qdiag = run["qdiag"]
        run_rows.append(
            "<tr>"
            f"<td>{html.escape(run['run_id'])}</td>"
            f"<td>{html.escape(run['stage'])}</td>"
            f"<td>{_fmt(run['corrected_tpot_mean_ms'])}</td>"
            f"<td>{_fmt(run['output_throughput_tokens_s'], 1)}</td>"
            f"<td>{_fmt(qdiag['gpu_rank_time_ms_per_output_token'], 6)}</td>"
            f"<td>{_fmt(qdiag['worker_cpu_rank_time_ms_per_output_token'], 6)}</td>"
            f"<td>{qdiag['samples']}</td>"
            "</tr>"
        )
    pair_rows = []
    for pair in pairs:
        pair_rows.append(
            "<tr>"
            f"<td>{pair['pair_index']}</td>"
            f"<td>{' → '.join(pair['run_order'])}</td>"
            f"<td>{_fmt(pair['corrected_tpot_delta_percent_hierarchical_vs_central'])}%</td>"
            f"<td>{_fmt(pair['output_throughput_delta_percent_hierarchical_vs_central'])}%</td>"
            f"<td>{_fmt(pair['gpu_rank_time_per_token_delta_percent'])}%</td>"
            f"<td>{_fmt(pair['worker_cpu_rank_time_per_token_delta_percent'])}%</td>"
            "</tr>"
        )
    decomposition_rows = []
    for run in runs:
        qdiag = run["qdiag"]
        decomposition_rows.append(
            "<tr>"
            f"<td>{html.escape(run['run_id'])}</td>"
            f"<td>{_fmt(_metric_mean(qdiag, 'stage_ms', 'ingress_drain_ms'))}</td>"
            f"<td>{_fmt(_metric_mean(qdiag, 'stage_ms', 'schedule_ms'))}</td>"
            f"<td>{_fmt(_metric_mean(qdiag, 'stage_ms', 'leader_arrival_skew_ms'))}</td>"
            f"<td>{_fmt(_metric_mean(qdiag, 'stage_ms', 'late_participant_collective_ms'))}</td>"
            f"<td>{_fmt(_metric_mean(qdiag, 'stage_ms', 'consensus_exposed_wait_ms'))}</td>"
            f"<td>{_fmt(_metric_mean(qdiag, 'stage_ms', 'consensus_overlap_window_ms'))}</td>"
            f"<td>{_fmt(_metric_mean(qdiag, 'stage_ms', 'leader_rendezvous_ms'))}</td>"
            f"<td>{_fmt(_metric_mean(qdiag, 'executor_ms', 'actor_submit_latency_ms', 'worker_command_submit_latency_ms'))}</td>"
            f"<td>{_fmt(_metric_mean(qdiag, 'executor_ms', 'send_seqs_latency_ms'))}</td>"
            f"<td>{_fmt(_metric_mean(qdiag, 'worker_ms', 'prepare_update_host_ms'))}</td>"
            f"<td>{_fmt(_metric_mean(qdiag, 'worker_ms', 'gpu_loop_ms'))}</td>"
            f"<td>{_fmt(_metric_mean(qdiag, 'worker_ms', 'token_materialize_ms'))}</td>"
            f"<td>{_fmt(qdiag['critical_worker_cpu_residual_ms']['mean'])}</td>"
            f"<td>{_fmt(_metric_mean(qdiag, 'executor_ms', 'result_rebuild_ms', 'result_unpack_ms'))}</td>"
            f"<td>{_fmt(_metric_mean(qdiag, 'stage_ms', 'postprocess_ms'))}</td>"
            "</tr>"
        )
    workload = comparison["validation"]["workload_totals"]
    diagnosis = html.escape(comparison["diagnosis"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Two-node quantum diagnostic A/B</title>
<style>
body{{font:15px/1.45 system-ui,sans-serif;margin:0;background:#0b1020;color:#e8edf8}}
main{{max-width:1180px;margin:auto;padding:32px}}h1{{margin-bottom:4px}}
.muted{{color:#9cabc8}}.card{{background:#151c30;border:1px solid #29334f;border-radius:12px;padding:18px;margin:18px 0}}
.verdict{{border-left:5px solid #66d9a5}}table{{border-collapse:collapse;width:100%}}
th,td{{padding:9px 10px;border-bottom:1px solid #29334f;text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
code{{color:#8bd5ff}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.kpi strong{{font-size:23px;display:block}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Two-node quantum diagnostic A/B</h1>
<p class="muted">Symmetric centralized/hierarchical diagnostics, normalized by actual output work.</p>
<div class="grid">
<div class="card kpi"><span>Prompt tokens</span><strong>{workload['prompt_tokens']:,}</strong></div>
<div class="card kpi"><span>Output tokens</span><strong>{workload['output_tokens']:,}</strong></div>
<div class="card kpi"><span>Successful requests</span><strong>{workload['successful_requests']:,}</strong></div>
</div>
<div class="card verdict"><h2>Current diagnosis</h2><p>{diagnosis}</p></div>
<div class="card"><h2>Runs</h2><table><thead><tr><th>Run</th><th>Stage</th><th>Corrected TPOT ms</th><th>Output tok/s</th><th>GPU rank time / output token</th><th>Worker CPU residual / output token</th><th>qdiag samples</th></tr></thead><tbody>{''.join(run_rows)}</tbody></table></div>
<div class="card"><h2>Adjacent paired deltas</h2><p class="muted">All deltas are hierarchical versus central.</p><table><thead><tr><th>Pair</th><th>Order</th><th>TPOT</th><th>Throughput</th><th>GPU rank time/token</th><th>Worker CPU residual/token</th></tr></thead><tbody>{''.join(pair_rows)}</tbody></table></div>
<div class="card"><h2>Timing decomposition per quantum</h2><p class="muted">Means in milliseconds. Ingress drain is the slowest leader per quantum. Leader arrival skew is measured before Gloo; late-participant collective time isolates the collective after the last leader arrives. Exposed wait is only the blocking tail after overlapped planning. Leader rendezvous is retained as a raw launch-to-completion span, not collective latency. Worker fields contain every rank sample; critical residual uses the slowest worker in each quantum.</p><table><thead><tr><th>Run</th><th>Ingress drain</th><th>Schedule</th><th>Leader arrival skew</th><th>Late-participant collective</th><th>Exposed coordination wait</th><th>Consensus overlap window</th><th>Leader rendezvous</th><th>Actor submit</th><th>Send seqs</th><th>Worker prepare</th><th>GPU loop</th><th>Materialize</th><th>Critical CPU residual</th><th>Result rebuild</th><th>Postprocess</th></tr></thead><tbody>{''.join(decomposition_rows)}</tbody></table></div>
<div class="card"><h2>Interpretation</h2><p><code>GPU rank time / output token</code> sums CUDA-event time over every worker rank, then divides by identical actual output tokens. Worker CPU residual is <code>worker_total_ms - gpu_loop_ms</code> per rank. These totals remain comparable when hierarchical engines execute different numbers of denser quantums.</p></div>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("manifest contains no runs")
    runs = [_load_run(row, manifest.parent) for row in rows]
    workload_signatures = {
        tuple(run["workload"].items()) for run in runs
    }
    workload_totals_match = len(workload_signatures) == 1
    if not workload_totals_match:
        raise ValueError(
            "runs do not have identical prompt/output/request totals"
        )
    pairs = _build_pairs(runs)
    comparison = {
        "schema_version": 1,
        "manifest": str(manifest),
        "validation": {
            "workload_totals_match": workload_totals_match,
            "workload_totals": {
                field: runs[0]["workload"][field]
                for field in (
                    "prompt_tokens",
                    "output_tokens",
                    "successful_requests",
                )
            },
        },
        "runs": runs,
        "pairs": pairs,
        "pair_means": {
            field: _mean_pair_metric(pairs, field)
            for field in (
                "corrected_tpot_delta_percent_hierarchical_vs_central",
                "output_throughput_delta_percent_hierarchical_vs_central",
                "gpu_rank_time_per_token_delta_percent",
                "worker_cpu_rank_time_per_token_delta_percent",
            )
        },
        "diagnosis": _diagnosis(pairs),
    }
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json
        else manifest.parent / "comparison.json"
    )
    output_html = (
        args.output_html.expanduser().resolve()
        if args.output_html
        else manifest.parent / "report.html"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_html.write_text(_render_html(comparison), encoding="utf-8")
    print(f"validated {len(runs)} runs and {len(pairs)} pairs")
    print(f"comparison: {output_json}")
    print(f"report: {output_html}")


if __name__ == "__main__":
    main()
