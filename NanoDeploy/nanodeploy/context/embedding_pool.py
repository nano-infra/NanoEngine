"""EmbeddingPool – GPU buffer for vision embeddings with slot-based management.

Mirrors the CacheContext pattern: pre-allocate a contiguous GPU buffer,
manage slots via alloc/free, register as RDMA MR via PeerAgent for
zero-copy transfer between Encoder and Prefill engines.

Two roles:
- **Encoder side (write)**: VisionEncoder writes embeddings into allocated
  slots; Prefill reads via RDMA.
- **Prefill side (read)**: Allocates a receive buffer; fetches embeddings
  from Encoder via RDMA before prefill forward.
"""

from __future__ import annotations

import dataclasses
import heapq
from typing import Any

import torch

from nanodeploy.logging import get_logger

logger = get_logger("embedding_pool")

_VISION_EMBED_BUFFER_ID = "vision_embed"


@dataclasses.dataclass
class EmbeddingPool:
    """Slot-based vision embedding buffer on GPU.

    Each slot holds a fixed-size region that can store the embeddings for
    one image (up to ``max_tokens_per_slot`` merged vision tokens).

    Parameters
    ----------
    num_slots : int
        Number of concurrent image embedding slots.
    max_tokens_per_slot : int
        Maximum merged vision tokens per slot (e.g. 2560 for 1344×1344).
    hidden_size : int
        Hidden dimension of vision embeddings (e.g. 3584 for Qwen3.5-35B).
    device : str
        CUDA device string.
    dtype : torch.dtype
        Data type for the buffer.
    """

    num_slots: int
    max_tokens_per_slot: int
    hidden_size: int
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    # Runtime state (set in __post_init__)
    buffer: torch.Tensor = dataclasses.field(init=False)
    _free_slots: list[int] = dataclasses.field(init=False)  # min-heap
    _slot_token_counts: dict[int, int] = dataclasses.field(init=False)

    # RDMA
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
        self._free_slots = list(range(self.num_slots))  # already a min-heap (sorted)
        self._slot_token_counts = {}
        logger.info(
            f"EmbeddingPool: {self.num_slots} slots × "
            f"{self.max_tokens_per_slot} tokens × {self.hidden_size} hidden, "
            f"buffer={self.buffer.shape}, "
            f"{self.buffer.nelement() * self.buffer.element_size() / 1e9:.2f} GB"
        )

    # ------------------------------------------------------------------
    # Slot management
    # ------------------------------------------------------------------

    @property
    def available_slots(self) -> int:
        return len(self._free_slots)

    def allocate(self, num_tokens: int) -> int:
        """Allocate a slot and return its index.

        Raises ``RuntimeError`` if no slots are available or
        ``num_tokens`` exceeds ``max_tokens_per_slot``.
        """
        if num_tokens > self.max_tokens_per_slot:
            raise RuntimeError(
                f"num_tokens={num_tokens} exceeds max_tokens_per_slot="
                f"{self.max_tokens_per_slot}"
            )
        if not self._free_slots:
            raise RuntimeError("EmbeddingPool: no free slots")
        slot_idx = heapq.heappop(self._free_slots)  # O(log n)
        self._slot_token_counts[slot_idx] = num_tokens
        return slot_idx

    def free(self, slot_idx: int) -> None:
        """Return a slot to the free pool."""
        if slot_idx in self._slot_token_counts:
            del self._slot_token_counts[slot_idx]
        if slot_idx not in self._free_slots:
            heapq.heappush(self._free_slots, slot_idx)  # O(log n)

    def free_many(self, slot_indices: list[int]) -> None:
        """Free multiple slots at once."""
        for idx in slot_indices:
            if idx in self._slot_token_counts:
                del self._slot_token_counts[idx]
            if idx not in self._free_slots:
                self._free_slots.append(idx)
        heapq.heapify(self._free_slots)  # O(n) single heapify vs n × O(log n) pushes

    # ------------------------------------------------------------------
    # Tensor access
    # ------------------------------------------------------------------

    def get_slot_tensor(self, slot_idx: int) -> torch.Tensor:
        """Return a view of the buffer for a specific slot.

        Shape: ``[num_tokens, hidden_size]`` where num_tokens is the
        actual count stored in this slot.
        """
        n = self._slot_token_counts.get(slot_idx, self.max_tokens_per_slot)
        return self.buffer[slot_idx, :n, :]

    def write_slot(self, slot_idx: int, embeddings: torch.Tensor) -> None:
        """Write embeddings into a slot.

        Args:
            slot_idx: Target slot index (must be allocated).
            embeddings: Tensor of shape ``[num_tokens, hidden_size]``.
        """
        n = embeddings.shape[0]
        self.buffer[slot_idx, :n, :] = embeddings.to(
            device=self.device, dtype=self.dtype
        )
        self._slot_token_counts[slot_idx] = n

    # ------------------------------------------------------------------
    # RDMA support
    # ------------------------------------------------------------------

    def slot_byte_offset(self, slot_idx: int) -> int:
        """Byte offset of a slot from the start of the buffer."""
        return (
            slot_idx * self.max_tokens_per_slot * self.hidden_size * self.dtype.itemsize
        )

    def slot_num_bytes(self, slot_idx: int) -> int:
        """Number of bytes actually used by a slot's valid tokens."""
        n = self._slot_token_counts.get(slot_idx, 0)
        return n * self.hidden_size * self.dtype.itemsize

    def register_mr(self, peer_agent) -> int:
        """Register the entire buffer as an RDMA memory region.

        Returns the MR handler.
        """
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


# ---------------------------------------------------------------------------
# Module-level singleton (one per worker process, same pattern as CacheContext)
# ---------------------------------------------------------------------------

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
