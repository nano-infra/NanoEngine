import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from dlengine.server.openai_server import build_app
from dlengine.utils.trace_merger import merge_trace_jsons_to_gzip
from fastapi.testclient import TestClient


def _write_trace(path: Path, rank: int) -> None:
    trace = {
        "schemaVersion": 1,
        "distributedInfo": {
            "rank": rank,
            "world_size": 2,
        },
        "traceEvents": [
            {
                "name": "compute",
                "ph": "X",
                "pid": 1,
                "tid": 2,
                "ts": 10 + rank,
                "dur": 3,
                "args": {},
            },
            {
                "name": "process_name",
                "ph": "M",
                "pid": 1,
                "tid": 0,
                "args": {"name": "worker"},
            },
            {
                "name": "process_labels",
                "ph": "M",
                "pid": "GPU",
                "tid": "stream",
                "args": {"labels": "GPU 0"},
            },
        ],
        "traceName": str(path),
    }
    path.write_text(json.dumps(trace), encoding="utf-8")


def _load_gzip_trace(data: bytes) -> dict:
    return json.loads(gzip.decompress(data))


def test_stream_merge_remaps_ranks_and_creates_perfetto_gzip(tmp_path):
    rank0 = tmp_path / "worker_rank_0.pt.trace.json"
    rank1 = tmp_path / "worker_rank_1.pt.trace.json"
    _write_trace(rank0, 0)
    _write_trace(rank1, 1)
    output = tmp_path / "profile_merged.trace.json.gz"

    result = merge_trace_jsons_to_gzip([rank1, rank0], output)

    assert result == output
    merged = _load_gzip_trace(output.read_bytes())
    events = merged["traceEvents"]
    assert len(events) == 6
    assert events[0]["pid"] == 1
    assert events[0]["tid"] == 2
    assert events[1]["args"]["name"] == "worker_rank0"
    assert events[2]["pid"] == "GPU_0"
    assert events[3]["pid"] == 100_000_001
    assert events[3]["tid"] == 100_000_002
    assert events[4]["args"]["name"] == "worker_rank1"
    assert events[5]["pid"] == "GPU_100000000"
    assert events[5]["tid"] == "stream_100000000"


class _ProfilerWorker:
    def __init__(self, trace_dir: Path):
        self.trace_dir = trace_dir
        self.calls = 0

    async def stop_profiler(self):
        self.calls += 1
        traces = sorted(str(path) for path in self.trace_dir.glob("*.pt.trace.json"))
        return {
            "ok": True,
            "status": "stopped",
            "workers": [
                {
                    "rank": rank,
                    "status": "stopped",
                    "trace_dir": str(self.trace_dir),
                    "trace_files": traces,
                }
                for rank in range(2)
            ],
        }


def test_stop_profiler_merge_returns_perfetto_gzip(tmp_path):
    _write_trace(tmp_path / "worker_rank_0.pt.trace.json", 0)
    _write_trace(tmp_path / "worker_rank_1.pt.trace.json", 1)
    worker = _ProfilerWorker(tmp_path)
    app = build_app(SimpleNamespace(worker=worker))

    with TestClient(app) as client:
        response = client.post("/stop_profiler", json={"merge": True})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/gzip"
    assert response.headers["x-dlengine-profiler-trace-count"] == "2"
    assert "attachment" in response.headers["content-disposition"]
    assert response.headers["content-disposition"].endswith(
        f'filename="{tmp_path.name}_merged.trace.json.gz"'
    )
    merged = _load_gzip_trace(response.content)
    assert len(merged["traceEvents"]) == 6


def test_stop_profiler_without_merge_preserves_json_response(tmp_path):
    _write_trace(tmp_path / "worker_rank_0.pt.trace.json", 0)
    _write_trace(tmp_path / "worker_rank_1.pt.trace.json", 1)
    worker = _ProfilerWorker(tmp_path)
    app = build_app(SimpleNamespace(worker=worker))

    with TestClient(app) as client:
        response = client.post("/stop_profiler")

    assert response.status_code == 200
    assert response.json()["status"] == "stopped"


def test_stop_profiler_rejects_non_boolean_merge(tmp_path):
    worker = _ProfilerWorker(tmp_path)
    app = build_app(SimpleNamespace(worker=worker))

    with TestClient(app) as client:
        response = client.post("/stop_profiler", json={"merge": "yes"})

    assert response.status_code == 400
    assert response.json()["error"] == "merge must be a boolean"
    assert worker.calls == 0
