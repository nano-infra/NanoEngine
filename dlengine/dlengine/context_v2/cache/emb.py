"""Embedding pool cache for vision embeddings.

EmbeddingPool owns the encoder-side GPU buffer used to publish vision
embeddings for prefill workers. Transfer mechanics (PeerAgent/RDMA reads) live
outside the pool; this module owns the buffer, slot accounting, and local memory
registration handle.
"""

from __future__ import annotations

import dataclasses
import heapq
from typing import Any

import torch

from dlengine.logging import get_logger

logger = get_logger("embedding_pool")

_VISION_EMBED_BUFFER_ID = "vision_embed"


@dataclasses.dataclass
class EmbeddingPool:
    """Slot-based vision embedding buffer on GPU."""

    num_slots: int
    max_tokens_per_slot: int
    hidden_size: int
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    buffer: torch.Tensor = dataclasses.field(init=False)
    _free_slots: list[int] = dataclasses.field(init=False)
    _slot_token_counts: dict[int, int] = dataclasses.field(init=False)
    _peer_agent: Any = dataclasses.field(default=None, init=False)
    _local_mr_handler: int | None = dataclasses.field(default=None, init=False)

    def __post_init__(self):
        self.buffer = torch.zeros(
            self.num_slots,
            self.max_tokens_per_slot,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )
        self._free_slots = list(range(self.num_slots))
        self._slot_token_counts = {}
        logger.info(
            f"EmbeddingPool: {self.num_slots} slots x "
            f"{self.max_tokens_per_slot} tokens x {self.hidden_size} hidden, "
            f"buffer={self.buffer.shape}, "
            f"{self.buffer.nelement() * self.buffer.element_size() / 1e9:.2f} GB"
        )

    @property
    def available_slots(self) -> int:
        return len(self._free_slots)

    def allocate(self, num_tokens: int) -> int:
        if num_tokens > self.max_tokens_per_slot:
            raise RuntimeError(
                f"num_tokens={num_tokens} exceeds max_tokens_per_slot="
                f"{self.max_tokens_per_slot}"
            )
        if not self._free_slots:
            raise RuntimeError("EmbeddingPool: no free slots")
        slot_idx = heapq.heappop(self._free_slots)
        self._slot_token_counts[slot_idx] = num_tokens
        return slot_idx

    def free(self, slot_idx: int) -> None:
        if slot_idx in self._slot_token_counts:
            del self._slot_token_counts[slot_idx]
        if slot_idx not in self._free_slots:
            heapq.heappush(self._free_slots, slot_idx)

    def free_many(self, slot_indices: list[int]) -> None:
        for idx in slot_indices:
            if idx in self._slot_token_counts:
                del self._slot_token_counts[idx]
            if idx not in self._free_slots:
                self._free_slots.append(idx)
        heapq.heapify(self._free_slots)

    def get_slot_tensor(self, slot_idx: int) -> torch.Tensor:
        n = self._slot_token_counts.get(slot_idx, self.max_tokens_per_slot)
        return self.buffer[slot_idx, :n, :]

    def write_slot(self, slot_idx: int, embeddings: torch.Tensor) -> None:
        n = embeddings.shape[0]
        self.buffer[slot_idx, :n, :] = embeddings.to(
            device=self.device, dtype=self.dtype
        )
        self._slot_token_counts[slot_idx] = n

    def slot_byte_offset(self, slot_idx: int) -> int:
        return (
            slot_idx * self.max_tokens_per_slot * self.hidden_size * self.dtype.itemsize
        )

    def slot_num_bytes(self, slot_idx: int) -> int:
        n = self._slot_token_counts.get(slot_idx, 0)
        return n * self.hidden_size * self.dtype.itemsize

    def register_mr(self, peer_agent) -> int:
        self._peer_agent = peer_agent
        buf_size = self.buffer.nelement() * self.buffer.element_size()
        self._local_mr_handler = peer_agent.register_memory_region(
            _VISION_EMBED_BUFFER_ID,
            self.buffer.data_ptr() + int(self.buffer.storage_offset()),
            buf_size,
        )
        logger.info(
            f"Registered vision embed MR: handler={self._local_mr_handler}, "
            f"size={buf_size / 1e6:.1f} MB"
        )
        return self._local_mr_handler


_EMBEDDING_POOL: EmbeddingPool | None = None


def get_embedding_pool() -> EmbeddingPool | None:
    return _EMBEDDING_POOL


def set_embedding_pool(
    num_slots: int,
    max_tokens_per_slot: int,
    hidden_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> EmbeddingPool:
    global _EMBEDDING_POOL
    _EMBEDDING_POOL = EmbeddingPool(
        num_slots=num_slots,
        max_tokens_per_slot=max_tokens_per_slot,
        hidden_size=hidden_size,
        device=device,
        dtype=dtype,
    )
    return _EMBEDDING_POOL


def reset_embedding_pool() -> None:
    global _EMBEDDING_POOL
    _EMBEDDING_POOL = None


__all__ = [
    "EmbeddingPool",
    "_VISION_EMBED_BUFFER_ID",
    "get_embedding_pool",
    "reset_embedding_pool",
    "set_embedding_pool",
]
