import gzip
import json
from pathlib import Path

from dlengine.utils.trace_merger import merge_trace_jsons_to_gzip


def _write_graph_trace(path: Path) -> None:
    events = [
        {
            "name": "cudaGraphLaunch",
            "cat": "cuda_runtime",
            "ph": "X",
            "pid": 1,
            "tid": 7,
            "ts": 0,
            "dur": 1,
            "args": {},
        }
    ]
    for stream in (10, 11, 12, 13):
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 0,
                "tid": stream,
                "args": {"name": f"stream {stream}"},
            }
        )
        for name in ("shared_gemm", "shared_activation"):
            events.append(
                {
                    "name": name,
                    "cat": "kernel",
                    "ph": "X",
                    "pid": 0,
                    "tid": stream,
                    "ts": stream * 10,
                    "dur": 3,
                    "args": {"stream": stream},
                }
            )
    events.append(
        {
            "name": "target_model",
            "cat": "kernel",
            "ph": "X",
            "pid": 0,
            "tid": 7,
            "ts": 1,
            "dur": 100,
            "args": {"stream": 7},
        }
    )
    path.write_text(
        json.dumps(
            {
                "distributedInfo": {"rank": 0, "world_size": 1},
                "traceEvents": events,
                "traceName": str(path),
            }
        )
    )


def _read(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def test_compacts_identical_graph_branches_without_touching_raw_trace(tmp_path):
    source = tmp_path / "worker_rank_0.pt.trace.json"
    regular = tmp_path / "regular.trace.json.gz"
    compact = tmp_path / "compact.trace.json.gz"
    _write_graph_trace(source)

    merge_trace_jsons_to_gzip([source], regular)
    merge_trace_jsons_to_gzip([source], compact, compact_streams=True)

    regular_kernels = [
        event for event in _read(regular)["traceEvents"] if event.get("cat") == "kernel"
    ]
    compact_events = _read(compact)["traceEvents"]
    compact_kernels = [
        event for event in compact_events if event.get("cat") == "kernel"
    ]

    assert {event["tid"] for event in regular_kernels} == {7, 10, 11, 12, 13}
    assert {event["tid"] for event in compact_kernels} == {7, 10}
    assert {event["args"]["stream"] for event in compact_kernels} == {7, 10}
    assert any(
        event.get("args", {})
        .get("name", "")
        .startswith("logical CUDA graph branch (4 lanes)")
        for event in compact_events
    )

    # Compaction is an output transform; the worker trace retains every lane.
    assert {
        event["tid"] for event in json.loads(source.read_text())["traceEvents"]
    } == {
        7,
        10,
        11,
        12,
        13,
    }
