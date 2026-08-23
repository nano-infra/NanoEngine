import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from dlengine.server.openai_server import build_app
from fastapi.testclient import TestClient


def _write_trace(path: Path) -> None:
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
    for stream in (20, 21, 22, 23):
        events.append(
            {
                "name": "branch",
                "cat": "kernel",
                "ph": "X",
                "pid": 0,
                "tid": stream,
                "ts": stream,
                "dur": 1,
                "args": {"stream": stream},
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


class _Worker:
    def __init__(self, trace: Path):
        self.trace = trace

    async def stop_profiler(self):
        return {
            "ok": True,
            "status": "stopped",
            "workers": [
                {
                    "rank": 0,
                    "status": "stopped",
                    "trace_dir": str(self.trace.parent),
                    "trace_files": [str(self.trace)],
                }
            ],
        }


def test_stop_profiler_can_return_compacted_trace(tmp_path):
    trace = tmp_path / "worker_rank_0.pt.trace.json"
    _write_trace(trace)
    app = build_app(SimpleNamespace(worker=_Worker(trace)))

    with TestClient(app) as client:
        response = client.post(
            "/stop_profiler",
            json={"merge": True, "compact_streams": True},
        )

    assert response.status_code == 200
    assert response.headers["x-dlengine-profiler-compact-streams"] == "true"
    assert "_compact_merged.trace.json.gz" in response.headers["content-disposition"]
    merged = json.loads(gzip.decompress(response.content))
    assert {
        event["tid"] for event in merged["traceEvents"] if event.get("cat") == "kernel"
    } == {20}


def test_compaction_requires_merge(tmp_path):
    trace = tmp_path / "worker_rank_0.pt.trace.json"
    _write_trace(trace)
    app = build_app(SimpleNamespace(worker=_Worker(trace)))

    with TestClient(app) as client:
        response = client.post("/stop_profiler", json={"compact_streams": True})

    assert response.status_code == 400
    assert response.json()["error"] == "compact_streams requires merge=true"
