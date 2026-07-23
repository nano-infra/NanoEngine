from __future__ import annotations

import json

from scripts.bench_ls_decode_snapshot_replay import (
    extract_snapshot,
    snapshot_to_layout,
)


def _log_line(timestamp: str, payload: dict) -> str:
    return f"[{timestamp}] [NANODEPLOY] [INFO] test.py:1 test - " f"{payload!r}\n"


def test_extract_snapshot_reconstructs_tokens_blocks_dop_and_pending(
    tmp_path,
) -> None:
    source_log = tmp_path / "serving.log"
    completion_jsonl = tmp_path / "completion.jsonl"

    admission = {
        "mode": "ls_decode_admission",
        "records": [
            {
                "sequence_id": 1,
                "dp_idx": 0,
                "planned_kv_dop": 1,
                "planned_kv_ranks": [0],
            },
            {
                "sequence_id": 2,
                "dp_idx": 1,
                "planned_kv_dop": 2,
                "planned_kv_ranks": [1, 2],
            },
        ],
    }
    prior = {
        "mode": "ls_decode_iteration",
        "iteration_sequence_ids": [[1], [2]],
        "iteration_master_assignments": [[1], [2]],
        "execution_loop_count": 16,
    }
    target = {
        "mode": "ls_decode_iteration",
        "group_ids": [10, 20],
        "group_dp_indices": [0, 1],
        "iteration_sequence_ids": [[1], [2]],
        "iteration_master_assignments": [[1], [1]],
        "group_used_kv_tokens": [
            [64, 16, 0, 0, 0, 0, 0, 0],
            [0, 60, 56, 0, 0, 0, 0, 0],
        ],
        "group_used_kv_blocks": [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
        ],
        "pending_append_blocks_per_master": [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ],
        "master_ranks": [[1], [1]],
        "master_batch_sizes": [[1], [1]],
        "kv_dops": [2, 2],
        "execution_loop_count": 16,
        "model_runner_duration_ms": 160.0,
        "step_itl_ms": 12.0,
        "planning_latency_ms": 3.0,
    }
    summary = {
        "mode": "decode",
        "total_batch_size": 2,
        "sp_size_hist_global": {2: 2},
        "sch_ovhd": "3.00ms",
        "post_sch_ovhd": "1.00ms",
        "max_kv_util_pct": "1.00",
        "min_free_blocks": 100,
    }
    source_log.write_text(
        "".join(
            [
                _log_line("2026-07-23 00:00:00", admission),
                _log_line("2026-07-23 00:00:01", prior),
                _log_line("2026-07-23 00:00:02", target),
                _log_line("2026-07-23 00:00:02", summary),
            ]
        ),
        encoding="utf-8",
    )
    completion_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"seq_id": 1, "prompt_len": 64}),
                json.dumps({"seq_id": 2, "prompt_len": 100}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    snapshot = extract_snapshot(
        source_log,
        completion_jsonl,
        target_iteration_index=1,
    )

    assert snapshot["validation"]["exact"] is True
    assert snapshot["target"]["dop_histogram"] == {"2": 2}
    assert snapshot["inferred_initial_multi_rank_requests"] == [
        {
            "sequence_id": 2,
            "prompt_len": 100,
            "planned_ranks": [1, 2],
            "inferred_prompt_tokens_by_sp": [0, 60, 40, 0, 0, 0, 0, 0],
        }
    ]
    by_id = {item["source_sequence_id"]: item for item in snapshot["sequences"]}
    assert by_id[1]["committed_tokens_by_sp"] == [64, 16, 0, 0, 0, 0, 0, 0]
    assert by_id[2]["committed_tokens_by_sp"] == [0, 60, 56, 0, 0, 0, 0, 0]

    layout = snapshot_to_layout(snapshot)
    assert [seq.context_len for seq in layout.sequences] == [80, 116]
    assert [seq.master_sp_idx for seq in layout.sequences] == [1, 1]
