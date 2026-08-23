"""Optional logical-lane compaction for CUDA Graph profiler traces."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


class StreamLaneCompactor:
    """Collapse repeated, identical CUDA Graph branch lanes for visualization.

    CUDA Graph replay can expose one CUPTI stream lane for every repeated
    fork/join branch even when the application supplied one reusable side
    stream. Lanes with identical kernel-name/count signatures are logical
    copies of the same branch role. Groups of at least four are mapped onto a
    single lane in the compacted output. Raw worker traces remain untouched.
    """

    def __init__(self, min_group_size: int = 4) -> None:
        if min_group_size < 2:
            raise ValueError("min_group_size must be at least 2")
        self.min_group_size = min_group_size
        self._graph_launch_count = 0
        self._kernels: dict[tuple[Any, Any], Counter[str]] = defaultdict(Counter)
        self._mapping: dict[tuple[Any, Any], Any] = {}
        self._group_sizes: dict[tuple[Any, Any], int] = {}
        self._emitted_thread_metadata: set[tuple[Any, Any]] = set()

    def observe(self, event: dict[str, Any]) -> None:
        if (
            event.get("cat") == "cuda_runtime"
            and event.get("name") == "cudaGraphLaunch"
        ):
            self._graph_launch_count += 1
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            return
        if "pid" not in event or "tid" not in event:
            return
        self._kernels[(event["pid"], event["tid"])][str(event.get("name", ""))] += 1

    def finalize(self) -> int:
        if self._graph_launch_count == 0:
            return 0
        groups: dict[tuple[Any, tuple[tuple[str, int], ...]], list[Any]] = defaultdict(
            list
        )
        for (pid, tid), kernels in self._kernels.items():
            signature = tuple(sorted(kernels.items()))
            groups[(pid, signature)].append(tid)

        compacted = 0
        for (pid, _signature), tids in groups.items():
            if len(tids) < self.min_group_size:
                continue
            target = min(tids) if all(type(tid) is int for tid in tids) else tids[0]
            self._group_sizes[(pid, target)] = len(tids)
            for tid in tids:
                self._mapping[(pid, tid)] = target
                compacted += tid != target
        return compacted

    def compact(self, event: dict[str, Any]) -> dict[str, Any] | None:
        key = (event.get("pid"), event.get("tid"))
        target = self._mapping.get(key)
        if target is None:
            return event

        old_tid = event["tid"]
        event["tid"] = target
        args = event.get("args")
        if isinstance(args, dict):
            if args.get("stream") == old_tid:
                args["stream"] = target
            if event.get("ph") == "M" and event.get("name") == "thread_name":
                metadata_key = (event.get("pid"), target)
                if metadata_key in self._emitted_thread_metadata:
                    return None
                self._emitted_thread_metadata.add(metadata_key)
                group_size = self._group_sizes[metadata_key]
                args["name"] = f"logical CUDA graph branch ({group_size} lanes)"
        return event

    @property
    def remapped_lane_count(self) -> int:
        return sum(source != target for (_pid, source), target in self._mapping.items())
