"""DSA top-k selection state.

``IndexerTopKState`` carries the indexer's selected top-k (logical + physical +
HiSparse) indices so they can be reused across consecutive "shared" attention
layers, across PP boundaries, and across MTP iterations. It is produced by the
DSA backend and threaded by the model / MTP / graph-capture runners.
"""

from dataclasses import dataclass

import torch


@dataclass
class IndexerTopKState:
    """TopK selection shared by consecutive DSA attention layers."""

    logical_indices: torch.Tensor | None = None
    physical_indices: torch.Tensor | None = None
    hisparse_indices: torch.Tensor | None = None
    source_layer: int | None = None

    def publish(
        self,
        layer_idx: int,
        logical_indices: torch.Tensor,
        physical_indices: torch.Tensor | None = None,
    ) -> None:
        self.logical_indices = logical_indices
        self.physical_indices = physical_indices
        self.hisparse_indices = None
        self.source_layer = layer_idx

    def publish_hisparse(self, layer_idx: int, hisparse_indices: torch.Tensor) -> None:
        """Cache a stable hot mapping for recurrent reuse of one physical layer."""
        if self.source_layer != layer_idx:
            raise RuntimeError(
                f"Cannot publish HiSparse mapping for layer {layer_idx} from "
                f"indexer source layer {self.source_layer}"
            )
        self.hisparse_indices = hisparse_indices

    def require_hisparse(
        self, layer_idx: int, num_tokens: int, topk: int
    ) -> torch.Tensor | None:
        """Return a reusable hot mapping only for the same physical layer."""
        indices = self.hisparse_indices
        if self.source_layer != layer_idx or indices is None:
            return None
        if tuple(indices.shape) != (num_tokens, topk):
            return None
        return indices

    def require(
        self,
        layer_idx: int,
        num_tokens: int,
        topk: int,
        *,
        require_physical: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logical = self.logical_indices
        if logical is None or self.source_layer is None:
            raise RuntimeError(
                f"Shared indexer layer {layer_idx} has no TopK from a preceding full layer"
            )
        expected = (num_tokens, topk)
        if tuple(logical.shape) != expected:
            raise RuntimeError(
                f"Shared indexer layer {layer_idx} received stale TopK shape "
                f"{tuple(logical.shape)} from layer {self.source_layer}; expected {expected}"
            )
        physical = self.physical_indices
        if require_physical and (physical is None or tuple(physical.shape) != expected):
            shape = None if physical is None else tuple(physical.shape)
            raise RuntimeError(
                f"Shared indexer layer {layer_idx} received invalid physical TopK "
                f"shape {shape} from layer {self.source_layer}; expected {expected}"
            )
        return logical, physical

    def select_rows(self, rows: torch.Tensor) -> "IndexerTopKState":
        """Return the per-request TopK rows selected from a packed draft extend.

        GLM recurrent MTP reuses the DSA selection produced for the last
        verified token. Prefill and draft-extend forwards contain more than
        one row per request, so the runner must carry only the row that seeded
        the first draft into the remaining recurrent iterations.
        """
        if self.logical_indices is None or self.source_layer is None:
            return IndexerTopKState()
        physical = self.physical_indices
        hisparse = self.hisparse_indices
        return IndexerTopKState(
            logical_indices=self.logical_indices.index_select(0, rows),
            physical_indices=(
                None if physical is None else physical.index_select(0, rows)
            ),
            hisparse_indices=(
                None if hisparse is None else hisparse.index_select(0, rows)
            ),
            source_layer=self.source_layer,
        )


# Backwards-compatible alias (the private name is imported widely today).
_IndexerTopKState = IndexerTopKState


__all__ = ["IndexerTopKState", "_IndexerTopKState"]
