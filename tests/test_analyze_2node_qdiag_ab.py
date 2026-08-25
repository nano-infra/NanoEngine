import csv
import importlib.util
import json
from pathlib import Path
import subprocess


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_analyzer_validates_workload_and_normalizes_rank_time(tmp_path):
    root = Path(__file__).resolve().parents[1]
    analyzer = root / "utils_analysis" / "analyze_2node_qdiag_ab.py"
    rows = []
    for index, (stage, arch, tpot, worker_ms) in enumerate(
        (
            ("central", "legacy_global", 60.0, 10.0),
            ("hierarchical", "hierarchical", 63.0, 11.0),
        ),
        start=1,
    ):
        run_id = f"{index:02d}_{stage}"
        request_path = tmp_path / f"{run_id}.requests.jsonl"
        quantum_path = tmp_path / f"{run_id}.quantum.jsonl"
        summary_path = tmp_path / f"{run_id}.summary.json"
        _write_jsonl(
            request_path,
            (
                {
                    "status": "FINISHED",
                    "prompt_tokens": 10,
                    "actual_output_tokens": 5,
                },
                {
                    "status": "FINISHED",
                    "prompt_tokens": 20,
                    "actual_output_tokens": 7,
                },
            ),
        )
        _write_jsonl(
            quantum_path,
            (
                {
                    "schema_version": 2,
                    "scheduler_arch": arch,
                    "engine_id": -1 if stage == "central" else 0,
                    "quantum_id": 0,
                    "execute_ms": 12.0,
                    "schedule_ms": 1.0,
                    "postprocess_ms": 0.5,
                    "useful_real_batch_size": 2,
                    "attention_work_tokens": 30,
                    "executor": {
                        "actor_submit_latency_ms": 1.0,
                        "send_seqs_latency_ms": 2.0,
                        "worker_rank_timings": [
                            {
                                "global_rank": 0,
                                "worker_total_ms": worker_ms,
                                "gpu_loop_ms": 8.0,
                                "prepare_update_host_ms": 1.0,
                                "token_materialize_ms": 0.5,
                            }
                        ],
                    },
                },
            ),
        )
        summary_path.write_text(
            json.dumps(
                {
                    "benchmark_runtime_s": 10.0,
                    "successful_requests": 2,
                    "failed_requests": 0,
                    "request_metrics_jsonl": str(request_path),
                    "quantum_diagnostics_jsonl": str(quantum_path),
                    "quantum_diagnostic_samples": 1,
                    "tpot_with_queue_ms": {"mean": tpot},
                }
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "run_index": run_id,
                "stage": stage,
                "scheduler_arch": arch,
                "status": "success",
                "exit_code": "0",
                "started_at_utc": "2026-08-25T00:00:00Z",
                "log_dir": str(tmp_path / run_id),
                "summary_json": str(summary_path),
            }
        )

    manifest = tmp_path / "run_manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    result = subprocess.run(
        ["python3", str(analyzer), "--manifest", str(manifest)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    comparison = json.loads(
        (tmp_path / "comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["validation"]["workload_totals_match"]
    assert comparison["validation"]["workload_totals"] == {
        "prompt_tokens": 30,
        "output_tokens": 12,
        "successful_requests": 2,
    }
    assert comparison["runs"][0]["qdiag"][
        "gpu_rank_time_ms_per_output_token"
    ] == 8.0 / 12.0
    assert comparison["runs"][0]["qdiag"][
        "worker_cpu_rank_time_ms_per_output_token"
    ] == 2.0 / 12.0
    assert comparison["runs"][1]["qdiag"][
        "worker_cpu_rank_time_ms_per_output_token"
    ] == 3.0 / 12.0
    assert comparison["pairs"][0][
        "corrected_tpot_delta_percent_hierarchical_vs_central"
    ] == 5.0
    report = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Two-node quantum diagnostic A/B" in report
    assert "GPU rank time / output token" in report


def test_analyzer_rejects_unpaired_run_order():
    root = Path(__file__).resolve().parents[1]
    path = root / "utils_analysis" / "analyze_2node_qdiag_ab.py"
    spec = importlib.util.spec_from_file_location("qdiag_analyzer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    try:
        module._build_pairs([{"run_id": "01", "stage": "central"}])
    except ValueError as exc:
        assert "even run count" in str(exc)
    else:
        raise AssertionError("odd run count was accepted")
