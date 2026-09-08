#!/usr/bin/env python3
"""Run the BS/GPU=128 centralized-versus-hierarchical CPU comparison.

The primary comparison deliberately uses the same CPU-only scheduler boundary
on both sides.  A case contains all waiting-request admission samples and
steady-state decode samples.  The result matrix expands each case into two
architectures and two phases:

    1/2/4 logical nodes x 4 SP scenarios x 2 architectures x 2 phases = 48

Passing ``--allow-modelled-topologies`` additionally permits 8/16/32 logical
nodes (up to 256 logical GPUs) for matching large logical-scale reports. Those
records are explicitly labelled as independent LocalScheduler models and are
not claims about a deployable 32-node cluster.

No Ray actor, GPU resource, CUDA context, transport, or model execution is
started by this script.  The hierarchical profiler's parallel values are an
ideal critical-path model (the maximum local scheduler cost), not measured
multi-host wall-clock latency.  Full transport/coordination measurements
belong in a separate appendix.
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.scheduler_overhead.profile_hierarchical_scheduler_scalability import (  # noqa: E402
    DEFAULT_MODEL,
    HierarchicalProfileCase,
    run_case as run_hierarchical_case,
)
from scripts.scheduler_overhead.profile_scheduler_scalability import (  # noqa: E402
    DEFAULT_SCENARIOS,
    ProfileCase,
    SCENARIOS,
    run_case as run_centralized_case,
)


SUPPORTED_LOGICAL_NODES = (1, 2, 4)
MODELLED_LOGICAL_NODES = (8, 16, 32)
BS_PER_GPU = 128
LOOP_COUNT = 16
SHORT_CONTEXT_LEN = 1_024
LONG_CONTEXT_LEN = 428_033
BLOCK_SIZE = 64


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {value!r}"
        ) from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return parsed


def _parse_scenarios(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(parsed) - set(SCENARIOS))
    if not parsed or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown or empty scenarios: {unknown or value!r}"
        )
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("scenarios must not contain duplicates")
    return parsed


def validate_matrix(
    logical_nodes: Sequence[int],
    scenarios: Sequence[str],
    batch_size_per_gpu: int,
    *,
    allow_modelled_topologies: bool = False,
) -> None:
    if not logical_nodes:
        raise ValueError("at least one logical node is required")
    allowed_nodes = set(SUPPORTED_LOGICAL_NODES)
    if allow_modelled_topologies:
        allowed_nodes.update(MODELLED_LOGICAL_NODES)
    unsupported = sorted(set(logical_nodes) - allowed_nodes)
    if unsupported:
        if allow_modelled_topologies:
            raise ValueError(
                "only logical node counts 1, 2, 4, 8, 16, and 32 are allowed; "
                f"got {unsupported}"
            )
        raise ValueError(
            "only complete production topologies 1, 2, and 4 are allowed; "
            f"got {unsupported} (pass --allow-modelled-topologies for 8/16/32)"
        )
    if batch_size_per_gpu != BS_PER_GPU:
        raise ValueError(
            f"this comparison is fixed at BS/GPU={BS_PER_GPU}; "
            f"got {batch_size_per_gpu}"
        )
    unknown = sorted(set(scenarios) - set(SCENARIOS))
    if not scenarios or unknown:
        raise ValueError(f"unknown or empty scenarios: {unknown or scenarios!r}")


def build_case_matrix(
    logical_nodes: Sequence[int] = SUPPORTED_LOGICAL_NODES,
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    batch_size_per_gpu: int = BS_PER_GPU,
    *,
    allow_modelled_topologies: bool = False,
) -> list[tuple[int, str]]:
    """Return the executable topology/scenario cases in stable order."""

    validate_matrix(
        logical_nodes,
        scenarios,
        batch_size_per_gpu,
        allow_modelled_topologies=allow_modelled_topologies,
    )
    return [
        (nodes, scenario_name)
        for nodes in logical_nodes
        for scenario_name in scenarios
    ]


def expected_result_cell_count(
    logical_nodes: Sequence[int], scenarios: Sequence[str]
) -> int:
    return len(logical_nodes) * len(scenarios) * 2 * 2


def _stats(record: dict[str, Any], key: str) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"record field {key!r} is not a summary")
    for required in ("count", "mean", "p99"):
        if required not in value:
            raise ValueError(f"summary {key!r} lacks {required!r}")
    return value


def _all_hierarchical_correct(record: dict[str, Any]) -> bool:
    profiles = record.get("per_engine_profiles")
    if not isinstance(profiles, list) or not profiles:
        return False
    return all(
        isinstance(profile.get("correctness"), dict)
        and all(profile["correctness"].values())
        for profile in profiles
    )


def _base_fields(
    *,
    logical_nodes: int,
    scenario: str,
    batch_size_per_gpu: int,
) -> dict[str, Any]:
    logical_gpus = logical_nodes * 8
    return {
        "logical_nodes": logical_nodes,
        "logical_gpus": logical_gpus,
        "batch_size_per_gpu": batch_size_per_gpu,
        "total_requests": logical_gpus * batch_size_per_gpu,
        "scenario": scenario,
        "loop_count": LOOP_COUNT,
        "topology_scope": (
            "complete_production_topology"
            if logical_nodes in SUPPORTED_LOGICAL_NODES
            else "logical_independent_local_scheduler_replica_model"
        ),
    }


def _comparison_record(
    *,
    logical_nodes: int,
    scenario: str,
    centralized: dict[str, Any],
    decentralized: dict[str, Any],
) -> dict[str, Any]:
    base = _base_fields(
        logical_nodes=logical_nodes,
        scenario=scenario,
        batch_size_per_gpu=BS_PER_GPU,
    )
    central_admission = {
        "mean_ms": centralized["admission_mean_ms"],
        "p99_ms": centralized["admission_p99_ms"],
        "count": centralized["admission_iterations"],
    }
    central_decode = {
        "mean_ms": centralized["mean_ms"],
        "p99_ms": centralized["p99_ms"],
        "count": centralized["measured_iterations"],
    }
    # Primary decentralized admission uses the same scheduler-only boundary
    # as centralized Scheduler.schedule(): local planned admission after
    # queue insertion. The full Router/control-plane path remains preserved as
    # a separate diagnostic field below.
    decentralized_admission = _stats(
        decentralized["admission_ms"], "scheduler_admission_critical_ms"
    )
    decentralized_control_plane_admission = _stats(
        decentralized["admission_ms"], "modelled_admission_critical_ms"
    )
    decentralized_decode = _stats(
        decentralized, "modelled_parallel_quantum_ms"
    )
    router_plan = _stats(
        decentralized["admission_ms"], "router_plan_receipt_ms"
    )
    local_commit = _stats(
        decentralized["admission_ms"], "local_commit_critical_ms"
    )
    aggregate_decode = _stats(
        decentralized, "aggregate_local_cpu_ms_per_global_quantum"
    )
    central_passed = (
        centralized["total_requests"] == base["total_requests"]
        and centralized["actual_sp1_requests"]
        + centralized["actual_sp8_requests"]
        == base["total_requests"]
        and centralized["actual_sp8_requests"]
        == centralized["expected_sp8_requests"]
    )
    decentralized_passed = (
        (
            decentralized["deployment_topology_supported"]
            or decentralized["topology_scope"]
            == "logical_independent_local_scheduler_replica_model"
        )
        and decentralized["total_requests"] == base["total_requests"]
        and decentralized["actual_sp1_requests"]
        + decentralized["actual_sp8_requests"]
        == base["total_requests"]
        and decentralized["actual_sp8_requests"]
        == decentralized["expected_sp8_requests"]
        and _all_hierarchical_correct(decentralized)
    )
    central_decode_mean = float(central_decode["mean_ms"])
    decentral_decode_mean = float(decentralized_decode["mean"])
    central_admission_mean = float(central_admission["mean_ms"])
    decentral_admission_mean = float(decentralized_admission["mean"])
    return {
        **base,
        "centralized_admission_mean_ms": central_admission_mean,
        "centralized_admission_p99_ms": central_admission["p99_ms"],
        "decentralized_admission_mean_ms": decentral_admission_mean,
        "decentralized_admission_p99_ms": decentralized_admission["p99"],
        "decentralized_control_plane_admission_mean_ms": (
            decentralized_control_plane_admission["mean"]
        ),
        "decentralized_control_plane_admission_p99_ms": (
            decentralized_control_plane_admission["p99"]
        ),
        "decentralized_router_plan_mean_ms": router_plan["mean"],
        "decentralized_local_commit_critical_mean_ms": local_commit["mean"],
        "centralized_decode_mean_ms": central_decode_mean,
        "centralized_decode_p99_ms": central_decode["p99_ms"],
        "centralized_decode_mean_ms_per_step": central_decode_mean / LOOP_COUNT,
        "centralized_decode_p99_ms_per_step": (
            float(central_decode["p99_ms"]) / LOOP_COUNT
        ),
        "decentralized_decode_mean_ms": decentral_decode_mean,
        "decentralized_decode_p99_ms": decentralized_decode["p99"],
        "decentralized_decode_mean_ms_per_step": (
            decentral_decode_mean / LOOP_COUNT
        ),
        "decentralized_decode_p99_ms_per_step": (
            float(decentralized_decode["p99"]) / LOOP_COUNT
        ),
        "decentralized_decode_aggregate_cpu_mean_ms": aggregate_decode["mean"],
        "decentralized_admission_over_centralized_ratio": (
            decentral_admission_mean / central_admission_mean
        ),
        "decentralized_decode_over_centralized_ratio": (
            decentral_decode_mean / central_decode_mean
        ),
        "centralized_admission_samples": central_admission["count"],
        "decentralized_admission_samples": decentralized_admission["count"],
        "centralized_decode_samples": central_decode["count"],
        "decentralized_decode_samples": decentralized_decode["count"],
        "centralized_case_passed": central_passed,
        "decentralized_case_passed": decentralized_passed,
        "admission_result_passed": central_passed and decentralized_passed,
        "decode_result_passed": central_passed and decentralized_passed,
        "centralized_raw": centralized,
        "decentralized_raw": decentralized,
    }


def result_cells_for_comparison(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one architecture comparison into its four report cells."""

    common = {
        key: comparison[key]
        for key in (
            "logical_nodes",
            "logical_gpus",
            "batch_size_per_gpu",
            "total_requests",
            "scenario",
            "loop_count",
        )
    }
    values = (
        (
            "centralized",
            "admission",
            comparison["centralized_admission_mean_ms"],
            comparison["centralized_admission_p99_ms"],
            None,
            comparison["centralized_admission_samples"],
        ),
        (
            "decentralized",
            "admission",
            comparison["decentralized_admission_mean_ms"],
            comparison["decentralized_admission_p99_ms"],
            None,
            comparison["decentralized_admission_samples"],
        ),
        (
            "centralized",
            "decode",
            comparison["centralized_decode_mean_ms"],
            comparison["centralized_decode_p99_ms"],
            comparison["centralized_decode_mean_ms_per_step"],
            comparison["centralized_decode_samples"],
        ),
        (
            "decentralized",
            "decode",
            comparison["decentralized_decode_mean_ms"],
            comparison["decentralized_decode_p99_ms"],
            comparison["decentralized_decode_mean_ms_per_step"],
            comparison["decentralized_decode_samples"],
        ),
    )
    cells = []
    for architecture, phase, mean_ms, p99_ms, mean_per_step, count in values:
        cells.append(
            {
                **common,
                "architecture": architecture,
                "phase": phase,
                "mean_ms": mean_ms,
                "p99_ms": p99_ms,
                "mean_ms_per_step": mean_per_step,
                "sample_count": count,
                "case_passed": (
                    comparison["admission_result_passed"]
                    if phase == "admission"
                    else comparison["decode_result_passed"]
                ),
            }
        )
    return cells


def _metadata(args: argparse.Namespace, *, case_count: int) -> dict[str, Any]:
    return {
        "benchmark": "nanodeploy-bs128-centralized-decentralized-scheduler",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "repo_root": str(REPO_ROOT),
        "gpus_per_logical_node": 8,
        "batch_size_per_gpu": BS_PER_GPU,
        "logical_nodes": list(args.logical_nodes),
        "topology_scope": {
            str(nodes): (
                "complete_production_topology"
                if nodes in SUPPORTED_LOGICAL_NODES
                else "logical_independent_local_scheduler_replica_model"
            )
            for nodes in args.logical_nodes
        },
        "scenarios": list(args.scenarios),
        "case_count": case_count,
        "result_cell_count": expected_result_cell_count(
            args.logical_nodes, args.scenarios
        ),
        "loop_count": LOOP_COUNT,
        "implementation": {
            "decentralized_admission_commit": (
                "LocalScheduler.commit_planned_sequences (native C++)"
            ),
            "decentralized_decode_scheduler": (
                "LocalScheduler.cpp_scheduler.schedule (native C++)"
            ),
            "python_contract_bookkeeping": "diagnostic-only",
        },
        "timed_scope": {
            "centralized": (
                "bulk waiting-request Scheduler.schedule() and steady-state "
                "decode Scheduler.schedule()"
            ),
            "decentralized": (
                "LocalScheduler planned admission after queue insertion and "
                "native Scheduler.schedule() steady-state decode; Router "
                "planning/receipt is retained as a separate diagnostic"
            ),
        },
        "excluded_scope": (
            "scheduler/Sequence construction, queue insertion, Ray/RDMA/ZMQ, "
            "ModelRunner, CUDA, GPU kernels, and result destruction"
        ),
        "decentralized_critical_path_definition": (
            "LocalScheduler harness executes serially; primary decentralized "
            "scheduler-only admission/decode critical metrics use max local "
            "cost and are modelled parallel paths, not measured multi-host "
            "wall time. Full Router control-plane admission remains in raw "
            "records as modelled_admission_critical_ms."
        ),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def _write_csv(path: Path, records: Iterable[dict[str, Any]]) -> None:
    rows = list(records)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (float, int)):
        return f"{value:.3f}"
    return html.escape(str(value))


def _render_html(
    *,
    metadata: dict[str, Any],
    comparisons: Sequence[dict[str, Any]],
    result_cells: Sequence[dict[str, Any]],
) -> str:
    passed = sum(bool(cell["case_passed"]) for cell in result_cells)
    total = len(result_cells)
    rows = []
    for record in comparisons:
        rows.append(
            "<tr>"
            f"<td>{record['logical_gpus']}</td>"
            f"<td>{html.escape(record['scenario'])}</td>"
            f"<td>{_fmt(record['centralized_admission_mean_ms'])} / "
            f"{_fmt(record['decentralized_admission_mean_ms'])}</td>"
            f"<td>{_fmt(record['centralized_decode_mean_ms'])} / "
            f"{_fmt(record['decentralized_decode_mean_ms'])}</td>"
            f"<td>{_fmt(record['centralized_decode_p99_ms'])} / "
            f"{_fmt(record['decentralized_decode_p99_ms'])}</td>"
            f"<td>{_fmt(record['centralized_decode_mean_ms_per_step'])} / "
            f"{_fmt(record['decentralized_decode_mean_ms_per_step'])}</td>"
            f"<td>{'PASS' if record['admission_result_passed'] and record['decode_result_passed'] else 'FAIL'}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BS/GPU=128 中心化/去中心化 Scheduler 对照</title>
<style>
body{{margin:0;background:#f4f7f9;color:#17212b;font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:28px 20px 52px}}
.hero,.card{{background:#fff;border:1px solid #d7e0e6;border-radius:12px;padding:18px 20px;margin-bottom:14px}}
.hero{{border-left:5px solid #075985}}h1{{margin:0 0 8px;font-size:28px}}h2{{margin:20px 0 8px;font-size:20px}}
.muted{{color:#5c6b76}}.ok{{color:#087f5b;font-weight:700}}table{{width:100%;border-collapse:collapse;margin:10px 0 15px}}
th,td{{border:1px solid #d7e0e6;padding:8px;text-align:left}}th{{background:#edf3f6}}code{{background:#eef3f6;padding:1px 4px;border-radius:4px}}
pre{{background:#15232d;color:#e8f1f5;padding:12px;border-radius:8px;overflow:auto;font-size:12px}}
</style></head><body><main>
<section class="hero"><h1>BS/GPU=128 Scheduler 性能对照</h1>
<p class="muted">中心化 vs 去中心化 · admission-only vs decode-only · 完整拓扑与逻辑规模模型</p>
<p class="ok">结果格：{passed}/{total} 通过（本次目标 {total}/{total}）。</p>
<p>主表使用对称 scheduler-only 边界：去中心化 admission 在队列构造之后只计 LocalScheduler planned commit，decode 只计原生 C++ Scheduler.schedule()；Router 控制面开销保留在 raw records，不是主表指标。</p>
<p>所有去中心化 critical path 均是 CPU-only harness 的 max(local) 建模值，不是跨节点 wall-clock；transport/Ray/consensus 不在本表。</p></section>
<section class="card"><h2>主表</h2>
<table><tr><th>GPUs</th><th>策略</th><th>Admission Mean 中心/去中心</th><th>Decode Mean 中心/去中心</th><th>Decode P99 中心/去中心</th><th>Decode Mean/step 中心/去中心</th><th>状态</th></tr>
{''.join(rows)}</table></section>
<section class="card"><h2>测试边界</h2>
<pre>BS/GPU=128；N = logical_gpus × 128；loop_count=16
Admission：waiting=N → 全部 admitted；主表不计构造、入队和 Router planning，仅计 LocalScheduler planned commit
Decode：running=N、waiting=0；主表仅计原生 C++ Scheduler.schedule()；warmup={metadata['arguments'].get('warmup_iterations', '—')}，测量={metadata['arguments'].get('iterations', '—')} quantum
中心化：Scheduler.schedule()
去中心化主表：max(local LocalScheduler scheduler-only path)；Router plan/receipt 全量开销见 JSON raw admission_ms.modelled_admission_critical_ms</pre>
<p class="muted">生成时间：{html.escape(metadata['created_at_utc'])} · 主机：{html.escape(metadata['hostname'])}</p></section>
<section class="card"><h2>原始产物</h2><p><code>comparison_bs128.json</code> 保存全部 raw record、分解指标和正确性 gate；<code>comparison_bs128.csv</code> 是上述 12 行 side-by-side 主表；<code>result_cells.csv</code> 是 48 个架构/阶段结果格。</p></section>
</main></body></html>"""


def run_benchmark(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    validate_matrix(
        args.logical_nodes,
        args.scenarios,
        args.batch_size_per_gpu,
        allow_modelled_topologies=args.allow_modelled_topologies,
    )
    # This runner is explicitly CPU-only.  The imported profilers do not start
    # CUDA, but clearing visibility makes the experiment contract explicit.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output_dir = args.output_dir or (
        REPO_ROOT
        / "bench_logs"
        / "scheduler_overhead"
        / "bs128_centralized_decentralized"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_case_matrix(
        args.logical_nodes,
        args.scenarios,
        args.batch_size_per_gpu,
        allow_modelled_topologies=args.allow_modelled_topologies,
    )
    comparisons: list[dict[str, Any]] = []
    for index, (logical_nodes, scenario_name) in enumerate(matrix, start=1):
        scenario = SCENARIOS[scenario_name]
        logical_gpus = logical_nodes * 8
        total_requests = logical_gpus * BS_PER_GPU
        print(
            f"[{index}/{len(matrix)}] nodes={logical_nodes} "
            f"gpus={logical_gpus} scenario={scenario_name} "
            f"requests={total_requests} centralized",
            flush=True,
        )
        central_case = ProfileCase(
            scenario=scenario,
            logical_nodes=logical_nodes,
            batch_size_per_gpu=BS_PER_GPU,
            short_context_len=SHORT_CONTEXT_LEN,
            long_context_len=LONG_CONTEXT_LEN,
            block_size=BLOCK_SIZE,
            loop_count=LOOP_COUNT,
            seed=args.seed,
        )
        centralized = run_centralized_case(
            central_case,
            warmup_iterations=args.warmup_iterations,
            measured_iterations=args.iterations,
            admission_iterations=args.admission_iterations,
        )
        gc.collect()
        print(
            f"[{index}/{len(matrix)}] nodes={logical_nodes} "
            f"gpus={logical_gpus} scenario={scenario_name} decentralized",
            flush=True,
        )
        hierarchical_case = HierarchicalProfileCase(
            scenario=scenario,
            logical_nodes=logical_nodes,
            batch_size_per_gpu=BS_PER_GPU,
            short_context_len=SHORT_CONTEXT_LEN,
            long_context_len=LONG_CONTEXT_LEN,
            block_size=BLOCK_SIZE,
            seed=args.seed,
        )
        decentralized = run_hierarchical_case(
            hierarchical_case,
            model=args.model,
            warmup_iterations=args.warmup_iterations,
            measured_iterations=args.iterations,
            admission_iterations=args.admission_iterations,
            admission_batch_size=total_requests,
        )
        comparisons.append(
            _comparison_record(
                logical_nodes=logical_nodes,
                scenario=scenario_name,
                centralized=centralized,
                decentralized=decentralized,
            )
        )
        gc.collect()

    result_cells = [
        cell
        for comparison in comparisons
        for cell in result_cells_for_comparison(comparison)
    ]
    metadata = _metadata(args, case_count=len(comparisons))
    passed = sum(bool(cell["case_passed"]) for cell in result_cells)
    payload = {
        "metadata": metadata,
        "summary": {
            "result_cells": len(result_cells),
            "passed_result_cells": passed,
            "failed_result_cells": len(result_cells) - passed,
            "all_passed": passed == len(result_cells),
        },
        "comparisons": comparisons,
        "result_cells": result_cells,
    }
    json_path = output_dir / "comparison_bs128.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "comparison_bs128.csv"
    _write_csv(csv_path, comparisons)
    cells_csv_path = output_dir / "result_cells.csv"
    _write_csv(cells_csv_path, result_cells)
    html_path = output_dir / "report.html"
    html_path.write_text(
        _render_html(
            metadata=metadata,
            comparisons=comparisons,
            result_cells=result_cells,
        ),
        encoding="utf-8",
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "# BS/GPU=128 centralized/decentralized scheduler comparison\n\n"
        f"Result cells: {passed}/{len(result_cells)} passed.\n\n"
        "This is a CPU-only scheduler comparison. The decentralized critical "
        "path is modelled from the maximum local cost; it is not measured "
        "multi-host wall time. See `report.html` for the summary and "
        "`comparison_bs128.json` for raw records.\n",
        encoding="utf-8",
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {cells_csv_path}")
    print(f"Wrote {html_path}")
    print(f"Result cells: {passed}/{len(result_cells)} passed")
    return json_path, csv_path, cells_csv_path, html_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--logical-nodes",
        type=_parse_positive_ints,
        default=SUPPORTED_LOGICAL_NODES,
        help="Complete production topology counts; only 1,2,4 are allowed.",
    )
    parser.add_argument(
        "--scenarios",
        type=_parse_scenarios,
        default=DEFAULT_SCENARIOS,
    )
    parser.add_argument("--batch-size-per-gpu", type=int, default=BS_PER_GPU)
    parser.add_argument(
        "--allow-modelled-topologies",
        action="store_true",
        help=(
            "Allow logical node counts 8,16,32. These are CPU-only independent "
            "LocalScheduler models, not deployable physical topologies."
        ),
    )
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--admission-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    try:
        validate_matrix(
            args.logical_nodes,
            args.scenarios,
            args.batch_size_per_gpu,
            allow_modelled_topologies=args.allow_modelled_topologies,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.warmup_iterations < 0:
        parser.error("--warmup-iterations must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.admission_iterations <= 0:
        parser.error("--admission-iterations must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
