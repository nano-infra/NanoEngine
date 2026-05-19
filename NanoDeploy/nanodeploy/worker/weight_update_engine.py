from __future__ import annotations

import logging
import pickle
import time
from typing import Any

import torch

from nanodeploy.context.weight import WeightContext
from nanodeploy.worker.pull_weights import pull_named_tensors_via_rdma
from nanodeploy.worker.weight_update import apply_named_tensors_in_place

logger = logging.getLogger("nanodeploy")


class WeightUpdateEngine:
    """Coordinates RDMA weight pulls, local weight storage, and model apply."""

    def __init__(self, model: torch.nn.Module, weight_context: WeightContext):
        self.model = model
        self.weight_context = weight_context

    def apply_named_tensors(
        self, named_tensors: dict[str, torch.Tensor], *, sync: bool = True
    ) -> dict[str, int]:
        return apply_named_tensors_in_place(self.model, named_tensors, sync=sync)

    def apply_context(self, *, sync: bool = True) -> dict[str, int]:
        return self.apply_named_tensors(self.weight_context.named_tensors(), sync=sync)

    def pull_into_context(
        self, manifest_blob: bytes, train_alias: str
    ) -> dict[str, Any]:
        peer_context = self.weight_context.peer_context
        if peer_context is None:
            raise RuntimeError("WeightContext PeerAgentContext is not initialized")

        manifest = pickle.loads(manifest_blob)
        t0 = time.monotonic()
        named, mr_names = pull_named_tensors_via_rdma(
            peer_context,
            train_alias,
            manifest,
        )
        pull_s = time.monotonic() - t0
        mr_names_by_tensor = {entry.name: entry.mr_name for entry in manifest.entries}
        self.weight_context.load(
            named,
            version=getattr(manifest, "version", None),
            mr_names_by_tensor=mr_names_by_tensor,
        )
        return {
            "version": getattr(manifest, "version", None),
            "n_tensors": len(getattr(manifest, "entries", [])),
            "pull_s": pull_s,
        }

    def pull_and_apply(self, manifest_blob: bytes, train_alias: str) -> dict[str, Any]:
        pull_stats = self.pull_into_context(manifest_blob, train_alias)
        t0 = time.monotonic()
        try:
            counts = self.apply_context()
            apply_s = time.monotonic() - t0
        finally:
            self.weight_context.clear(release_mrs=True)

        stats = {**pull_stats, "apply_s": apply_s, **counts}
        logger.info("WeightUpdateEngine.pull_and_apply: %s", stats)
        return stats
