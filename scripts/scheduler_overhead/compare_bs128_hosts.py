#!/usr/bin/env python3
"""Compare two completed BS/GPU=128 scheduler runs from different hosts.

The input runs must have been produced by
``profile_bs128_centralized_decentralized.py`` with the same matrix.  The
output keeps each host's raw values and reports ``host_b / host_a`` ratios;
it never mixes old rjob1 data into the comparison.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRICS = (
    "centralized_admission_mean_ms",
    "decentralized_admission_mean_ms",
    "centralized_decode_mean_ms",
    "decentralized_decode_mean_ms",
    "centralized_decode_p99_ms",
    "decentralized_decode_p99_ms",
)
METRIC_LABELS = {
    "centralized_admission_mean_ms": "Admission Mean / centralized",
    "decentralized_admission_mean_ms": "Admission Mean / decentralized",
    "centralized_decode_mean_ms": "Decode Mean / centralized",
    "decentralized_decode_mean_ms": "Decode Mean / decentralized",
    "centralized_decode_p99_ms": "Decode P99 / centralized",
    "decentralized_decode_p99_ms": "Decode P99 / decentralized",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-a", type=Path, required=True)
    parser.add_argument("--host-b", type=Path, required=True)
    parser.add_argument("--label-a", default="rjob0")
    parser.add_argument("--label-b", default="rjob3")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load(path: Path) -> dict[str, Any]:
    source = path / "comparison_bs128.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    data = json.loads(source.read_text())
    if not data.get("summary", {}).get("all_passed", False):
        raise ValueError(f"input run is not fully passing: {source}")
    if not isinstance(data.get("comparisons"), list):
        raise ValueError(f"missing comparisons in {source}")
    return data


def _key(record: dict[str, Any]) -> tuple[int, str]:
    return int(record["logical_nodes"]), str(record["scenario"])


def _validate(a: dict[str, Any], b: dict[str, Any]) -> None:
    ma, mb = a["metadata"], b["metadata"]
    for field in ("batch_size_per_gpu", "loop_count", "scenarios"):
        if ma.get(field) != mb.get(field):
            raise ValueError(f"input mismatch for metadata field {field!r}")
    if ma.get("batch_size_per_gpu") != 128:
        raise ValueError("host comparison is fixed at BS/GPU=128")
    ca = {_key(item) for item in a["comparisons"]}
    cb = {_key(item) for item in b["comparisons"]}
    if ca != cb:
        raise ValueError(f"input matrices differ: {sorted(ca ^ cb)}")
    for record in a["comparisons"] + b["comparisons"]:
        if int(record["logical_nodes"]) == 3:
            raise ValueError("3-node results are excluded from this report")


def _rows(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    left = {_key(item): item for item in a["comparisons"]}
    right = {_key(item): item for item in b["comparisons"]}
    rows: list[dict[str, Any]] = []
    for key in sorted(left):
        ra, rb = left[key], right[key]
        row: dict[str, Any] = {
            "logical_nodes": key[0],
            "logical_gpus": int(ra["logical_gpus"]),
            "scenario": key[1],
            "topology_scope": ra.get("topology_scope", ""),
            "total_requests": int(ra["total_requests"]),
        }
        for metric in METRICS:
            va, vb = float(ra[metric]), float(rb[metric])
            row[f"{metric}_{a['metadata']['hostname']}"] = va
            row[f"{metric}_{b['metadata']['hostname']}"] = vb
            row[f"{metric}_ratio_b_over_a"] = vb / va if va else None
            row[f"{metric}_delta_pct_b_over_a"] = (vb / va - 1.0) * 100 if va else None
        rows.append(row)
    return rows


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scopes: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        scopes.setdefault(str(row["topology_scope"]), []).append(row)
    result = []
    for scope, scope_rows in scopes.items():
        summary: dict[str, Any] = {
            "topology_scope": scope,
            "row_count": len(scope_rows),
            "logical_nodes": sorted({row["logical_nodes"] for row in scope_rows}),
        }
        for metric in METRICS:
            ratios = [
                row[f"{metric}_ratio_b_over_a"]
                for row in scope_rows
                if row[f"{metric}_ratio_b_over_a"] is not None
            ]
            summary[metric] = {
                "label": METRIC_LABELS[metric],
                "median_ratio_b_over_a": statistics.median(ratios),
                "min_ratio_b_over_a": min(ratios),
                "max_ratio_b_over_a": max(ratios),
                "median_delta_pct_b_over_a": (statistics.median(ratios) - 1.0) * 100,
            }
        result.append(summary)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.3f}"


def _write_html(path: Path, a: dict[str, Any], b: dict[str, Any], rows: list[dict[str, Any]], summaries: list[dict[str, Any]], label_a: str, label_b: str) -> None:
    host_a, host_b = a["metadata"]["hostname"], b["metadata"]["hostname"]
    cards = []
    for summary in summaries:
        cards.append(f"<h3>{html.escape(summary['topology_scope'])}</h3><table><tr><th>指标</th><th>中位数 {label_b}/{label_a}</th><th>范围</th></tr>")
        for metric in METRICS:
            item = summary[metric]
            cards.append(
                f"<tr><td>{html.escape(METRIC_LABELS[metric])}</td>"
                f"<td>{item['median_ratio_b_over_a']:.3f}× ({item['median_delta_pct_b_over_a']:+.1f}%)</td>"
                f"<td>{item['min_ratio_b_over_a']:.3f}–{item['max_ratio_b_over_a']:.3f}×</td></tr>"
            )
        cards.append("</table>")
    table = [
        "<table><tr><th>GPUs</th><th>策略</th><th>范围</th>"
        f"<th>Admission Mean {label_a}/{label_b} ms</th>"
        f"<th>Decode Mean {label_a}/{label_b} ms</th>"
        f"<th>Decode P99 {label_a}/{label_b} ms</th>"
        f"<th>Decode Mean {label_b}/{label_a}</th></tr>"
    ]
    for row in rows:
        vals = []
        for metric in ("centralized_admission_mean_ms", "decentralized_admission_mean_ms"):
            vals.append(f"{_fmt(row[f'{metric}_{host_a}'])}/{_fmt(row[f'{metric}_{host_b}'])}")
        admission = "<br>".join(vals)
        decode = "<br>".join(
            f"{_fmt(row[f'{metric}_{host_a}'])}/{_fmt(row[f'{metric}_{host_b}'])}"
            for metric in ("centralized_decode_mean_ms", "decentralized_decode_mean_ms")
        )
        p99 = "<br>".join(
            f"{_fmt(row[f'{metric}_{host_a}'])}/{_fmt(row[f'{metric}_{host_b}'])}"
            for metric in ("centralized_decode_p99_ms", "decentralized_decode_p99_ms")
        )
        decode_ratio = "<br>".join(
            f"{row[f'{metric}_ratio_b_over_a']:.3f}×"
            for metric in ("centralized_decode_mean_ms", "decentralized_decode_mean_ms")
        )
        table.append(
            f"<tr><td>{row['logical_gpus']}</td><td>{html.escape(row['scenario'])}</td>"
            f"<td>{html.escape(row['topology_scope'])}</td><td>{admission}</td>"
            f"<td>{decode}</td><td>{p99}</td><td>{decode_ratio}</td></tr>"
        )
    table.append("</table>")
    generated = datetime.now(timezone.utc).isoformat()
    body = f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>BS/GPU=128 rjob0/rjob3 host comparison</title><style>
body{{margin:0;background:#f4f7f9;color:#17212b;font:14px/1.5 system-ui,sans-serif}}main{{max-width:1450px;margin:auto;padding:26px 18px 50px}}section{{background:#fff;border:1px solid #d7e0e6;border-radius:12px;padding:18px 20px;margin-bottom:14px}}h1{{margin:0 0 8px}}h2{{margin:20px 0 8px}}h3{{margin:14px 0 6px}}.ok{{color:#087f5b;font-weight:700}}.muted{{color:#5c6b76}}table{{width:100%;border-collapse:collapse;margin:8px 0 15px}}th,td{{border:1px solid #d7e0e6;padding:7px;text-align:left;vertical-align:top}}th{{background:#edf3f6}}code{{background:#eef3f6;padding:1px 4px;border-radius:4px}}.nowrap{{white-space:nowrap}}
</style></head><body><main><section><h1>BS/GPU=128：rjob0 ↔ rjob3 主机对比</h1><p class=\"ok\">输入矩阵全部通过；对比行：{len(rows)}。</p><p>比值定义：<code>{html.escape(label_b)}/{html.escape(label_a)}</code>。小于 1 表示 rjob3 更快，大于 1 表示 rjob3 更慢。</p><p class=\"muted\">{html.escape(label_a)}：{html.escape(host_a)}<br>{html.escape(label_b)}：{html.escape(host_b)}<br>生成时间：{generated}</p></section><section><h2>逐策略结果</h2>{''.join(table)}<p class=\"muted\">每个单元格按 centralized / decentralized 展示；Decode 原始量包含 loop_count=16，比较两主机时比例不受该共同因子影响。</p></section><section><h2>按拓扑范围的比值摘要</h2>{''.join(cards)}</section><section><h2>解释边界</h2><p>1/2/4 logical nodes 是完整 production topology CPU 模型；32 logical nodes = 256 GPUs 是 independent LocalScheduler replica 的逻辑规模模型，不是 32 节点真实部署 wall-clock。去中心化指标为模型化并行 critical path，未包含 Ray/RDMA/ZMQ、GPU kernel 或跨主机 transport。</p></section></main></body></html>"""
    path.write_text(body)


def main() -> int:
    args = _parser().parse_args()
    a, b = _load(args.host_a), _load(args.host_b)
    _validate(a, b)
    rows, summaries = _rows(a, b), None
    summaries = _summaries(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "host_comparison.csv", rows)
    payload = {
        "benchmark": "nanodeploy-bs128-centralized-decentralized-host-comparison",
        "comparison": {"host_a": args.label_a, "host_b": args.label_b, "ratio": "host_b / host_a"},
        "inputs": {
            "host_a": str(args.host_a),
            "host_b": str(args.host_b),
            "hostname_a": a["metadata"]["hostname"],
            "hostname_b": b["metadata"]["hostname"],
            "summary_a": a["summary"],
            "summary_b": b["summary"],
        },
        "rows": rows,
        "summaries": summaries,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_host": platform.node(),
    }
    (args.output_dir / "host_comparison.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    _write_html(args.output_dir / "report.html", a, b, rows, summaries, args.label_a, args.label_b)
    readme = f"""# BS/GPU=128 rjob0/rjob3 host comparison

Inputs: `{args.host_a}` and `{args.host_b}`. Both matrices must pass before comparison.

The ratio is **{args.label_b} / {args.label_a}**: below 1 means rjob3 is faster. See `report.html` for the readable report, `host_comparison.csv` for rows, and `host_comparison.json` for the complete machine-readable comparison.

The 1/2/4-node records are complete production-topology CPU models. The 256-GPU record uses 32 logical nodes and is explicitly a logical independent LocalScheduler replica model; it is not a real 32-node wall-clock measurement. Decentralized values are modelled parallel critical paths and exclude Ray/RDMA/ZMQ, GPU kernels, and transport.
"""
    (args.output_dir / "README.md").write_text(readme)
    print(f"Wrote {args.output_dir / 'host_comparison.json'}")
    print(f"Compared {len(rows)} rows; all input result cells passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
