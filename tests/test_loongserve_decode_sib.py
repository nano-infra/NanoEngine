import importlib.util
import json
import sys
from pathlib import Path


def _load_sib():
    path = Path(__file__).resolve().parents[1] / "docs-dev" / "loongserve_sp8_decode_sib.py"
    spec = importlib.util.spec_from_file_location("loongserve_sp8_decode_sib", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sib = _load_sib()


def _row(*, d, p90, viable=True, node_local=True, b=16, w=65536, l=4096):
    return {
        "profile_kind": "uniform",
        "B": b,
        "W_attn": w,
        "L_p90": l,
        "L_max": l,
        "d_attn": d,
        "viable": viable,
        "node_local": node_local,
        "step_p90_ms": p90,
    }


def test_build_sib_keeps_d1_when_it_is_near_optimal():
    rows = [
        _row(d=1, p90=10.0),
        _row(d=2, p90=9.8),
        _row(d=4, p90=9.7),
        _row(d=8, p90=10.2),
    ]

    entries = sib.build_sib(
        rows,
        include_dataset_replay=False,
        metric="step_p90_ms",
        near_optimal_ratio=1.05,
        abs_gain_ms=0.1,
        min_repeats=1,
        stability_ratio=1.05,
    )

    assert len(entries) == 1
    assert entries[0]["d_target"] == 1
    assert entries[0]["d_near"] == 1
    assert entries[0]["threshold_reason"] == "d=1_remains_near_optimal"


def test_build_sib_chooses_fastest_memory_forced_viable_d():
    rows = [
        _row(d=1, p90=100.0, viable=False),
        _row(d=2, p90=20.0),
        _row(d=4, p90=15.0),
        _row(d=8, p90=12.0),
    ]

    entries = sib.build_sib(
        rows,
        include_dataset_replay=False,
        metric="step_p90_ms",
        near_optimal_ratio=1.05,
        abs_gain_ms=0.1,
        min_repeats=1,
        stability_ratio=1.05,
    )

    assert entries[0]["d_mem"] == 2
    assert entries[0]["best_d"] == 8
    assert entries[0]["d_target"] == 8
    assert entries[0]["performance_threshold"] == "d2_to_d8"


def test_build_sib_filters_non_node_local_rows():
    rows = [
        _row(d=1, p90=10.0),
        _row(d=2, p90=9.0, node_local=False),
    ]

    entries = sib.build_sib(
        rows,
        include_dataset_replay=False,
        metric="step_p90_ms",
        near_optimal_ratio=1.05,
        abs_gain_ms=0.1,
        min_repeats=1,
        stability_ratio=1.05,
    )

    assert entries[0]["metric_by_d"] == {"1": 10.0}
    assert entries[0]["d_target"] == 1


def test_sib_cli_writes_json_and_markdown(tmp_path, capsys):
    profile = tmp_path / "profile.jsonl"
    rows = [_row(d=1, p90=10.0), _row(d=2, p90=8.0)]
    profile.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out_json = tmp_path / "sib.json"
    out_md = tmp_path / "thresholds.md"

    old_argv = sys.argv
    try:
        sys.argv = [
            "loongserve_sp8_decode_sib.py",
            str(profile),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--min-repeats",
            "1",
        ]
        sib.main()
    finally:
        sys.argv = old_argv

    payload = json.loads(out_json.read_text())
    assert payload["kind"] == "loongserve_sp8_decode_sib"
    assert payload["entries"][0]["d_target"] == 2
    assert "LoongServe SP8 Decode Thresholds" in out_md.read_text()
    assert "entries: 1" in capsys.readouterr().out
