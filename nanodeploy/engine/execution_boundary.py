from __future__ import annotations

import threading
from collections.abc import Mapping


class ExecutionBoundaryRecorder:
    """Accumulates low-overhead executor boundary timings."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._sample_count = 0
            self._totals: dict[str, float] = {}
            self._maxima: dict[str, float] = {}
            self._minima: dict[str, float] = {}

    def record(self, metrics: Mapping[str, float]) -> None:
        with self._lock:
            self._sample_count += 1
            for name, raw_value in metrics.items():
                value = float(raw_value)
                self._totals[name] = (
                    self._totals.get(name, 0.0) + value
                )
                self._maxima[name] = max(
                    self._maxima.get(name, value),
                    value,
                )
                self._minima[name] = min(
                    self._minima.get(name, value),
                    value,
                )

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            metrics: dict[str, float | int] = {
                "sample_count": self._sample_count,
            }
            for name, total in self._totals.items():
                metrics[f"{name}_total"] = total
                metrics[f"{name}_mean"] = (
                    total / self._sample_count
                    if self._sample_count
                    else 0.0
                )
                metrics[f"{name}_min"] = self._minima[name]
                metrics[f"{name}_max"] = self._maxima[name]
            return metrics
