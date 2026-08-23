"""Merge distributed PyTorch profiler traces without loading them into memory.

The rank remapping follows DLSLime's json_merger.py:

* numeric process/thread ids are offset by rank * 100_000_000;
* string process/thread ids receive a rank suffix;
* process and thread metadata names receive a rank suffix.

Unlike the original implementation, events are decoded and written one at a
time. This keeps memory bounded for multi-gigabyte production traces.

Adapted from DLSLime, which in turn adapts Triton-distributed's trace merger.
DLSLime source revision: 1442385 (MIT).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TextIO

from dlengine.utils.trace_compactor import StreamLaneCompactor

logger = logging.getLogger(__name__)

_RANK_PID_STRIDE = 100_000_000
_TRACE_EVENTS_RE = re.compile(r'"traceEvents"\s*:\s*\[')
_DISTRIBUTED_RANK_RE = re.compile(
    r'"distributedInfo"\s*:\s*\{.*?"rank"\s*:\s*(\d+)',
    re.DOTALL,
)
_FILENAME_RANK_RE = re.compile(r"(?:^|[_-])rank[_-]?(\d+)(?:[._-]|$)")
_READ_CHUNK_CHARS = 1024 * 1024
_MAX_EVENT_CHARS = 256 * 1024 * 1024


class _TraceEventStream:
    """Incrementally decode one trace's top-level traceEvents array."""

    def __init__(self, path: Path):
        self.path = path
        self._file: TextIO | None = None
        self._buffer = ""
        self._pos = 0
        self._eof = False
        self.prefix = ""
        self.suffix = ""

    def __enter__(self) -> "_TraceEventStream":
        self._file = self.path.open(
            "r",
            encoding="utf-8",
            errors="replace",
            buffering=_READ_CHUNK_CHARS,
        )
        self.prefix = self._read_prefix()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._file is not None:
            self._file.close()

    def _read_prefix(self) -> str:
        assert self._file is not None
        chunks: list[str] = []
        total = 0
        while True:
            chunk = self._file.read(64 * 1024)
            if not chunk:
                raise ValueError(f"{self.path}: missing traceEvents array")
            chunks.append(chunk)
            total += len(chunk)
            prefix = "".join(chunks)
            match = _TRACE_EVENTS_RE.search(prefix)
            if match is not None:
                self._buffer = prefix[match.end() :]
                return prefix[: match.end()]
            if total > 64 * 1024 * 1024:
                raise ValueError(f"{self.path}: trace header exceeds 64 MiB")

    def _read_more(self) -> bool:
        assert self._file is not None
        if self._pos:
            self._buffer = self._buffer[self._pos :]
            self._pos = 0
        chunk = self._file.read(_READ_CHUNK_CHARS)
        if not chunk:
            self._eof = True
            return False
        self._buffer += chunk
        return True

    def __iter__(self) -> Iterator[dict[str, Any]]:
        assert self._file is not None
        decoder = json.JSONDecoder(strict=False)
        while True:
            while (
                self._pos < len(self._buffer) and self._buffer[self._pos] in " \t\r\n,"
            ):
                self._pos += 1

            if self._pos < len(self._buffer) and self._buffer[self._pos] == "]":
                self._pos += 1
                self.suffix = self._buffer[self._pos :] + self._file.read()
                return

            if self._pos >= len(self._buffer):
                if self._read_more():
                    continue
                raise ValueError(f"{self.path}: unterminated traceEvents array")

            try:
                event, end = decoder.raw_decode(self._buffer, self._pos)
            except json.JSONDecodeError as error:
                pending = len(self._buffer) - self._pos
                if not self._eof and pending <= _MAX_EVENT_CHARS and self._read_more():
                    continue
                raise ValueError(
                    f"{self.path}: invalid trace event near character {error.pos}"
                ) from error

            if not isinstance(event, dict):
                raise ValueError(f"{self.path}: trace event must be a JSON object")
            self._pos = end
            yield event


def _rank_for_trace(path: Path, prefix: str) -> int:
    match = _FILENAME_RANK_RE.search(path.name)
    if match is not None:
        return int(match.group(1))
    match = _DISTRIBUTED_RANK_RE.search(prefix)
    if match is not None:
        return int(match.group(1))
    raise ValueError(f"{path}: cannot determine distributed rank")


def _remap_identifier(value: Any, delta: int) -> Any:
    if isinstance(value, str):
        return f"{value}_{delta}"
    if type(value) is int:
        return value + delta
    raise ValueError(f"unsupported profiler identifier {value!r}")


def _remap_event(event: dict[str, Any], rank: int) -> dict[str, Any]:
    delta = rank * _RANK_PID_STRIDE
    if "pid" in event:
        event["pid"] = _remap_identifier(event["pid"], delta)
    if "tid" in event:
        event["tid"] = _remap_identifier(event["tid"], delta)

    if event.get("ph") == "M":
        args = event.get("args")
        if isinstance(args, dict):
            if event.get("name") in {"process_name", "thread_name"}:
                name = args.get("name")
                if name is not None:
                    args["name"] = f"{name}_rank{rank}"
            elif event.get("name") == "process_labels":
                labels = args.get("labels")
                if labels is not None:
                    args["labels"] = f"{labels}_{rank}"
    return event


def _trace_sort_key(path: Path) -> tuple[int, str]:
    match = _FILENAME_RANK_RE.search(path.name)
    return (int(match.group(1)) if match is not None else 2**31, path.name)


def merge_trace_jsons(
    trace_files: Sequence[str | Path],
    output_json: str | Path,
    *,
    compact_streams: bool = False,
) -> Path:
    """Stream multiple profiler traces into one Chrome/Perfetto JSON file."""
    inputs = sorted(
        (Path(path).resolve() for path in trace_files),
        key=_trace_sort_key,
    )
    if not inputs:
        raise ValueError("at least one profiler trace is required")
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing profiler traces: {missing}")

    output = Path(output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_ranks: set[int] = set()
    first_suffix: str | None = None
    event_count = 0

    if output.name.endswith(".gz"):
        destination_context = gzip.open(
            output,
            "wt",
            encoding="utf-8",
            compresslevel=3,
        )
    else:
        destination_context = output.open(
            "w",
            encoding="utf-8",
            buffering=4 * 1024 * 1024,
        )

    with destination_context as destination:
        first_event = True
        for index, path in enumerate(inputs):
            logger.info("Merging profiler trace %s", path)
            compactor = None
            if compact_streams:
                compactor = StreamLaneCompactor()
                with _TraceEventStream(path) as compact_scan:
                    for compact_event in compact_scan:
                        compactor.observe(compact_event)
                compacted_lanes = compactor.finalize()
                logger.info(
                    "Compacting %s repeated CUDA graph stream lanes in %s",
                    compacted_lanes,
                    path,
                )
            with _TraceEventStream(path) as stream:
                rank = _rank_for_trace(path, stream.prefix)
                if rank in seen_ranks:
                    raise ValueError(f"duplicate profiler rank {rank}: {path}")
                seen_ranks.add(rank)
                if index == 0:
                    destination.write(stream.prefix)

                for event in stream:
                    if compactor is not None:
                        event = compactor.compact(event)
                        if event is None:
                            continue
                    if not first_event:
                        destination.write(",")
                    json.dump(
                        _remap_event(event, rank),
                        destination,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    first_event = False
                    event_count += 1
                    if event_count % 250_000 == 0:
                        logger.info("Merged %s profiler events", f"{event_count:,}")

                if index == 0:
                    first_suffix = stream.suffix

        if first_suffix is None:
            raise ValueError("first profiler trace did not provide a JSON suffix")
        first_suffix = re.sub(
            r'("traceName"\s*:\s*)".*?"',
            lambda match: match.group(1) + json.dumps(str(output)),
            first_suffix,
            count=1,
            flags=re.DOTALL,
        )
        destination.write("]")
        destination.write(first_suffix)

    logger.info(
        "Merged %d traces and %s events into %s",
        len(inputs),
        f"{event_count:,}",
        output,
    )
    return output


def merge_trace_jsons_to_gzip(
    trace_files: Sequence[str | Path],
    output_trace: str | Path,
    *,
    compact_streams: bool = False,
) -> Path:
    """Create a Perfetto-compatible gzip-compressed trace JSON."""
    output = Path(output_trace).resolve()
    if not output.name.endswith(".trace.json.gz"):
        output = output.with_name(output.name.removesuffix(".gz") + ".trace.json.gz")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + ".tmp.gz")
    try:
        merge_trace_jsons(
            trace_files,
            temporary_output,
            compact_streams=compact_streams,
        )
        temporary_output.replace(output)
    finally:
        temporary_output.unlink(missing_ok=True)

    logger.info("Created merged profiler trace %s", output)
    return output


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stream-merge distributed PyTorch profiler traces into one "
            "Perfetto-compatible .trace.json.gz"
        ),
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--compact-streams",
        action="store_true",
        help="collapse repeated CUDA Graph branch lanes in the merged output",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    inputs = sorted(
        args.input_dir.rglob("*.pt.trace.json"),
        key=_trace_sort_key,
    )
    output = args.output or (
        args.input_dir / f"{args.input_dir.name}_merged.trace.json.gz"
    )
    result = merge_trace_jsons_to_gzip(
        inputs, output, compact_streams=args.compact_streams
    )
    print(result)


if __name__ == "__main__":
    _main()
