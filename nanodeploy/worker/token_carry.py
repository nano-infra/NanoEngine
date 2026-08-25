from __future__ import annotations

from collections.abc import Sequence

import torch


TokenCarryKey = tuple[int, int]


class GpuTokenCarry:
    """Keep one quantum of sampled tokens addressable by request identity."""

    def __init__(self) -> None:
        self._tokens: dict[TokenCarryKey, torch.Tensor] = {}
        self._wave_id: int | None = None
        self._quantum_id: int | None = None

    @staticmethod
    def _validate_rows(
        token_ids: torch.Tensor,
        keys: Sequence[TokenCarryKey],
    ) -> tuple[TokenCarryKey, ...]:
        frozen_keys = tuple(keys)
        if token_ids.ndim != 1:
            raise ValueError("GPU token carry requires one-dimensional token IDs")
        if token_ids.numel() != len(frozen_keys):
            raise ValueError(
                "GPU token carry row count does not match request keys"
            )
        if len(set(frozen_keys)) != len(frozen_keys):
            raise ValueError("GPU token carry request keys must be unique")
        return frozen_keys

    def select_inputs(
        self,
        payload_token_ids: torch.Tensor,
        keys: Sequence[TokenCarryKey],
        *,
        wave_id: int,
        quantum_id: int,
    ) -> tuple[torch.Tensor, int]:
        """Select previous GPU tokens for a consecutive same-epoch request."""
        frozen_keys = self._validate_rows(payload_token_ids, keys)
        if (
            self._wave_id != wave_id
            or self._quantum_id is None
            or self._quantum_id + 1 != quantum_id
        ):
            return payload_token_ids, 0

        selected: list[torch.Tensor] = []
        hits = 0
        for row, key in enumerate(frozen_keys):
            carried = self._tokens.get(key)
            if carried is None:
                selected.append(payload_token_ids[row])
                continue
            if (
                carried.device != payload_token_ids.device
                or carried.dtype != payload_token_ids.dtype
            ):
                raise RuntimeError(
                    "GPU token carry device or dtype changed between quantums"
                )
            selected.append(carried)
            hits += 1
        if hits == 0:
            return payload_token_ids, 0
        return torch.stack(selected), hits

    def record(
        self,
        sampled_token_ids: torch.Tensor,
        keys: Sequence[TokenCarryKey],
        *,
        wave_id: int,
        quantum_id: int,
    ) -> None:
        """Replace the carry set after one successfully launched quantum."""
        frozen_keys = self._validate_rows(sampled_token_ids, keys)
        self._tokens = {
            key: sampled_token_ids[row].detach()
            for row, key in enumerate(frozen_keys)
        }
        self._wave_id = wave_id
        self._quantum_id = quantum_id

    def reset(self) -> None:
        self._tokens.clear()
        self._wave_id = None
        self._quantum_id = None
