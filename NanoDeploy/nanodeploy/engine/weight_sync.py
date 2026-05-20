from __future__ import annotations

import threading
import time
from contextlib import contextmanager


class WeightUpdateBarrier:
    """Coordinate weight sync with generation / streaming scheduler progress.

    ``generation`` / ``update`` protect the legacy blocking ``generate`` path.
    ``update_streaming`` is used by the continuous rollout path: it pauses
    admission, waits for active running work to drain while the rollout loop
    continues calling ``step_once``, then lets the caller apply weights.
    """

    def __init__(self):
        self._cv = threading.Condition()
        self._updating = False
        self._active_generations = 0

    @contextmanager
    def generation(self):
        with self._cv:
            while self._updating:
                self._cv.wait()
            self._active_generations += 1
        try:
            yield
        finally:
            with self._cv:
                self._active_generations -= 1
                if self._active_generations == 0:
                    self._cv.notify_all()

    @contextmanager
    def update(self):
        t0 = time.monotonic()
        with self._cv:
            self._updating = True
            while self._active_generations > 0:
                self._cv.wait()
            wait_s = time.monotonic() - t0
        try:
            yield wait_s
        finally:
            with self._cv:
                self._updating = False
                self._cv.notify_all()

    @contextmanager
    def update_streaming(self, engine):
        t0 = time.monotonic()
        engine.pause_admission()
        with self._cv:
            self._updating = True
            while engine.num_active_for_update() > 0:
                self._cv.wait(timeout=0.5)
            wait_s = time.monotonic() - t0
        try:
            yield wait_s
        finally:
            engine.resume_admission()
            with self._cv:
                self._updating = False
                self._cv.notify_all()

    def notify_step(self) -> None:
        with self._cv:
            self._cv.notify_all()

    @property
    def active_generations(self) -> int:
        with self._cv:
            return self._active_generations

    @property
    def updating(self) -> bool:
        with self._cv:
            return self._updating
