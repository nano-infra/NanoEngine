from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Mapping

import ray

from nanodeploy.engine.hierarchical_contract import (
    CoordinatorStatus,
    StartWave,
)


@dataclass(slots=True)
class _RegisteredEngine:
    fingerprint: str
    ready: bool = False


class DecodeCoordinatorState:
    """Pure deployment-wide wave state; it never owns requests or KV state."""

    def __init__(self, expected_engines: int) -> None:
        if expected_engines <= 1:
            raise ValueError(
                "DecodeCoordinator is only required when attention_dp > 1"
            )
        self.expected_engines = expected_engines
        self.wave_id = 0
        self.running = False
        self.pending_wakeup = False
        self._engines: dict[int, _RegisteredEngine] = {}

    @property
    def ready(self) -> bool:
        return (
            len(self._engines) == self.expected_engines
            and all(engine.ready for engine in self._engines.values())
        )

    def register(self, engine_id: int, fingerprint: str) -> None:
        if not 0 <= engine_id < self.expected_engines:
            raise ValueError(f"engine_id {engine_id} is outside deployment")
        if not fingerprint:
            raise ValueError("config fingerprint must not be empty")
        existing = self._engines.get(engine_id)
        if existing is not None:
            if existing.fingerprint != fingerprint:
                raise RuntimeError(
                    f"engine {engine_id} changed collective fingerprint"
                )
            return
        self._engines[engine_id] = _RegisteredEngine(fingerprint=fingerprint)

    def mark_ready(self, engine_id: int) -> None:
        try:
            engine = self._engines[engine_id]
        except KeyError as exc:
            raise RuntimeError(
                f"engine {engine_id} was not registered"
            ) from exc
        fingerprints = {
            registered.fingerprint for registered in self._engines.values()
        }
        if len(fingerprints) != 1:
            raise RuntimeError(
                "collective-sensitive config fingerprint mismatch"
            )
        engine.ready = True

    def first_request(
        self, target_engine_id: int, observed_wave_id: int
    ) -> StartWave | None:
        if not self.ready:
            raise RuntimeError("deployment is not READY")
        if target_engine_id not in self._engines:
            raise ValueError(
                f"target engine {target_engine_id} is not registered"
            )
        if observed_wave_id > self.wave_id:
            raise ValueError(
                "FIRST_REQ observed a future wave: "
                f"observed={observed_wave_id}, current={self.wave_id}"
            )
        if self.running:
            # This deliberately favors liveness over avoiding one possible
            # empty follow-up wave: an ADD racing with WAVE_COMPLETE cannot be
            # allowed to remain asleep.
            self.pending_wakeup = True
            return None
        self.wave_id += 1
        self.running = True
        self.pending_wakeup = False
        return StartWave(self.wave_id)

    def wave_complete(self, wave_id: int) -> StartWave | None:
        if not self.running:
            raise RuntimeError("WAVE_COMPLETE received while paused")
        if wave_id != self.wave_id:
            raise ValueError(
                f"stale WAVE_COMPLETE: got={wave_id}, current={self.wave_id}"
            )
        self.running = False
        if not self.pending_wakeup:
            return None
        self.pending_wakeup = False
        self.wave_id += 1
        self.running = True
        return StartWave(self.wave_id)

    def status(self) -> CoordinatorStatus:
        return CoordinatorStatus(
            wave_id=self.wave_id,
            running=self.running,
            ready=self.ready,
            pending_wakeup=self.pending_wakeup,
        )


@ray.remote(num_cpus=0.1, max_concurrency=16)
class DecodeCoordinator:
    """Ray actor that broadcasts wave transitions to LocalEngineCore actors."""

    def __init__(self, expected_engines: int) -> None:
        self._state = DecodeCoordinatorState(expected_engines)
        self._engines: dict[int, Any] = {}
        self._lock = threading.Lock()

    def attach_engines(self, engines: Mapping[int, Any]) -> None:
        with self._lock:
            if set(engines) != set(range(self._state.expected_engines)):
                raise ValueError("coordinator engine registry is incomplete")
            self._engines = dict(engines)

    def register(self, engine_id: int, fingerprint: str) -> None:
        with self._lock:
            self._state.register(engine_id, fingerprint)

    def mark_ready(self, engine_id: int) -> CoordinatorStatus:
        with self._lock:
            self._state.mark_ready(engine_id)
            return self._state.status()

    def _broadcast(self, start: StartWave | None) -> None:
        if start is None:
            return
        if len(self._engines) != self._state.expected_engines:
            raise RuntimeError("coordinator has no complete engine registry")
        for engine in self._engines.values():
            engine.start_wave.remote(start.wave_id)

    def first_request(
        self, target_engine_id: int, observed_wave_id: int
    ) -> CoordinatorStatus:
        with self._lock:
            start = self._state.first_request(
                target_engine_id, observed_wave_id
            )
            self._broadcast(start)
            return self._state.status()

    def wave_complete(self, wave_id: int) -> CoordinatorStatus:
        with self._lock:
            start = self._state.wave_complete(wave_id)
            self._broadcast(start)
            return self._state.status()

    def status(self) -> CoordinatorStatus:
        with self._lock:
            return self._state.status()
