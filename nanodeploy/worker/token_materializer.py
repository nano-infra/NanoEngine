from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class PendingTokenMaterialization:
    slot: int
    host_token_ids: torch.Tensor = field(repr=False)
    device_token_ids: torch.Tensor = field(repr=False)
    source_ready_event: Any = field(repr=False)
    copy_done_event: Any = field(repr=False)
    submitted_at: float
    materializer_id: int = field(repr=False, compare=False)


class AsyncTokenMaterializer:
    """D2H token copies with one pinned host buffer per transport slot."""

    def __init__(self, *, num_slots: int) -> None:
        if num_slots <= 0:
            raise ValueError("token materializer requires at least one slot")
        self.num_slots = num_slots
        self._host_buffers: list[torch.Tensor | None] = [None] * num_slots
        self._active: dict[int, PendingTokenMaterialization] = {}
        self._copy_stream: Any = None

    def _host_buffer_for(self, token_ids: torch.Tensor, slot: int) -> torch.Tensor:
        host = self._host_buffers[slot]
        if (
            host is None
            or host.shape != token_ids.shape
            or host.dtype != token_ids.dtype
        ):
            host = torch.empty_like(
                token_ids,
                device="cpu",
                pin_memory=True,
            )
            self._host_buffers[slot] = host
        return host

    def submit(
        self,
        token_ids: torch.Tensor,
        *,
        slot: int,
    ) -> PendingTokenMaterialization:
        if not 0 <= slot < self.num_slots:
            raise ValueError(f"invalid token materialization slot {slot}")
        if slot in self._active:
            raise RuntimeError(
                f"token materialization slot {slot} is still active"
            )
        if token_ids.ndim != 2:
            raise ValueError(
                "token materialization requires a two-dimensional token matrix"
            )

        device_token_ids = token_ids.detach()
        source_ready_event = None
        copy_done_event = None
        if device_token_ids.is_cuda:
            host_token_ids = self._host_buffer_for(device_token_ids, slot)
            if self._copy_stream is None:
                self._copy_stream = torch.cuda.Stream(
                    device=device_token_ids.device
                )
            source_ready_event = torch.cuda.Event()
            copy_done_event = torch.cuda.Event()
            source_ready_event.record(
                torch.cuda.current_stream(device=device_token_ids.device)
            )
            with torch.cuda.stream(self._copy_stream):
                self._copy_stream.wait_event(source_ready_event)
                host_token_ids.copy_(device_token_ids, non_blocking=True)
                copy_done_event.record(self._copy_stream)
        else:
            host_token_ids = device_token_ids.to(device="cpu", copy=True)

        pending = PendingTokenMaterialization(
            slot=slot,
            host_token_ids=host_token_ids,
            device_token_ids=device_token_ids,
            source_ready_event=source_ready_event,
            copy_done_event=copy_done_event,
            submitted_at=time.perf_counter(),
            materializer_id=id(self),
        )
        self._active[slot] = pending
        return pending

    def collect(
        self,
        pending: PendingTokenMaterialization,
    ) -> list[list[int]]:
        if pending.materializer_id != id(self):
            raise ValueError(
                "token materialization belongs to another materializer"
            )
        if self._active.get(pending.slot) is not pending:
            raise RuntimeError("token materialization is not active")
        if pending.copy_done_event is not None:
            pending.copy_done_event.synchronize()
        token_rows = pending.host_token_ids.tolist()
        del self._active[pending.slot]
        return token_rows
