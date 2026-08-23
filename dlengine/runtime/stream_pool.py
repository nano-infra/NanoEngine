"""Process-local CUDA streams shared by sequential model layers.

Model layers execute serially, so giving every layer its own side stream does
not add concurrency. It only retains one CUDA stream per layer and makes
profiler timelines unnecessarily wide. This pool keeps distinct streams for
operations that can overlap within a layer, while reusing each role across
layers and across target/MTP model instances on the same device.
"""

from __future__ import annotations

import os
import threading

import torch

_STREAMS: dict[tuple[int, int, str, int], torch.cuda.Stream] = {}
_STREAMS_LOCK = threading.Lock()


def _cuda_device_index(device: torch.device | str | int | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    normalized = torch.device(device)
    if normalized.type != "cuda":
        raise ValueError(f"CUDA stream requested for non-CUDA device {device!r}")
    return (
        normalized.index
        if normalized.index is not None
        else torch.cuda.current_device()
    )


def get_cuda_stream(
    role: str,
    device: torch.device | str | int | None = None,
    *,
    priority: int = 0,
) -> torch.cuda.Stream:
    """Return the shared stream for the process, device, role and priority.

    Callers must use different roles for streams that need to run concurrently
    (for example attention_q and attention_kv). Reusing a role across layers is
    safe because every call site joins its side stream before the next decoder
    layer starts.
    """
    if not role:
        raise ValueError("CUDA stream role must be non-empty")
    device_index = _cuda_device_index(device)
    key = (os.getpid(), device_index, role, priority)
    with _STREAMS_LOCK:
        stream = _STREAMS.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=device_index, priority=priority)
            _STREAMS[key] = stream
        return stream


def _clear_cuda_stream_pool_for_test() -> None:
    """Drop cached references; only for unit tests with mocked streams."""
    with _STREAMS_LOCK:
        _STREAMS.clear()
