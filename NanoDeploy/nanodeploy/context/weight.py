from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass
from typing import Any

import torch

from nanodeploy.context.peer_agent import PeerAgentContext

logger = logging.getLogger("nanodeploy")

_DTYPE_TO_STR = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


@dataclass
class WeightTensorEntry:
    name: str
    tensor: torch.Tensor
    mr_name: str | None = None
    version: int | None = None


class WeightContext:
    """Versioned local store for tensors pulled by the weight-update path."""

    def __init__(self, peer_context: PeerAgentContext | None = None):
        self.peer_context = peer_context
        self.version: int | None = None
        self.entries: dict[str, WeightTensorEntry] = {}
        self._registered_mrs: list[str] = []

    def set_peer_agent_context(self, peer_context: PeerAgentContext | None) -> None:
        self.peer_context = peer_context

    def put(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        mr_name: str | None = None,
        version: int | None = None,
    ) -> None:
        entry_version = self.version if version is None else version
        self.entries[name] = WeightTensorEntry(name, tensor, mr_name, entry_version)
        if mr_name is not None:
            self._registered_mrs.append(mr_name)

    def load(
        self,
        named_tensors: dict[str, torch.Tensor],
        *,
        version: int | None = None,
        mr_names_by_tensor: dict[str, str] | None = None,
    ) -> None:
        self.clear(release_mrs=True)
        self.version = version
        mr_names_by_tensor = mr_names_by_tensor or {}
        for name, tensor in named_tensors.items():
            self.put(
                name,
                tensor,
                mr_name=mr_names_by_tensor.get(name),
                version=version,
            )

    def named_tensors(self) -> dict[str, torch.Tensor]:
        return {name: entry.tensor for name, entry in self.entries.items()}

    def pull_named_tensors_via_rdma(self, train_alias: str, manifest) -> None:
        """Pull every manifest entry into this context via RDMA.

        The receive buffers are CPU pinned tensors. Their local memory regions
        are tracked in this context and released by ``clear(release_mrs=True)``
        after the tensors have been applied to model parameters.
        """
        if self.peer_context is None:
            raise RuntimeError("WeightContext PeerAgentContext is not initialized")

        self.peer_context.ensure_connected(train_alias, timeout=60.0)
        peer_agent = self.peer_context.agent
        endpoint = peer_agent._get_endpoint(train_alias)
        conn = peer_agent._get_connection(train_alias)

        received: dict[str, torch.Tensor] = {}
        mr_names_by_tensor: dict[str, str] = {}
        assigns: list[tuple[int, int, int, int, int]] = []

        for entry in manifest.entries:
            dtype = _STR_TO_DTYPE[entry.dtype]
            buf = torch.empty(
                tuple(entry.shape), dtype=dtype, device="cpu", pin_memory=True
            )
            received[entry.name] = buf
            mr_names_by_tensor[entry.name] = entry.mr_name

            peer_agent.register_memory_region(
                entry.mr_name, buf.data_ptr(), 0, entry.size
            )
            local_handle = peer_agent.get_handle(
                entry.mr_name, resource_key=conn.local_key
            )
            remote_handle = peer_agent.get_handle(
                entry.mr_name,
                train_alias,
                resource_key=conn.peer_key,
                endpoint=endpoint,
            )
            # endpoint.read expects (local_handle, remote_handle, remote_offset,
            # local_offset, length) — see dlslime/peer_agent/_agent.py:1487.
            assigns.append((local_handle, remote_handle, 0, 0, entry.size))

        slot = endpoint.read(assigns, None)
        slot.wait()
        self.load(
            received,
            version=getattr(manifest, "version", None),
            mr_names_by_tensor=mr_names_by_tensor,
        )

    def release_mrs(self) -> None:
        if self.peer_context is None:
            self._registered_mrs.clear()
            return

        for mr_name in self._registered_mrs:
            try:
                self.peer_context.unregister_memory_region(mr_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("unregister_memory_region(%s) failed: %s", mr_name, exc)
        self._registered_mrs.clear()

    def clear(self, *, release_mrs: bool = True) -> None:
        if release_mrs:
            self.release_mrs()
        else:
            self._registered_mrs.clear()
        self.entries.clear()
        self.version = None


def apply_named_tensors_in_place(
    model: torch.nn.Module,
    named_tensors: dict[str, torch.Tensor],
    *,
    sync: bool = True,
) -> dict[str, int]:
    """Apply HF-named full tensors to matching model parameters in place.

    Delegates to ``model.load_weights(...)`` when available so that
    model-specific name remapping (HF ``q/k/v_proj`` → fused ``qkv_proj``,
    ``gate_proj/up_proj`` → ``gate_up_proj``) and the packed-loader
    signature (``weight_loader(param, tensor, shard_id, weight_name)``)
    are handled by the same code path that loads the initial checkpoint.

    Falls back to the previous direct-lookup behaviour only for models
    that don't implement ``load_weights`` (e.g. test/dummy models). The
    fallback intentionally counts misses so callers still see the diag
    stats; this is unsafe for fused-weight models — log a warning when
    that path is hit on a real model.

    All copies are *in place* (the loaders use ``param.data.copy_``)
    so any captured CUDA graphs continue to point at the same allocation.
    """
    if hasattr(model, "load_weights"):
        # Model-aware path: per-model loader (e.g. qwen3_loader) does
        # the HF→internal name mapping + calls weight_loader with
        # (param, tensor, shard_id, weight_name) for fused weights.
        # Wrap the dict as the (weight_name, raw_weight_name, tensor)
        # generator the loaders expect — same shape used by the safetensors
        # checkpoint path.
        weights_iter = ((name, name, t) for name, t in named_tensors.items())
        try:
            model.load_weights(weights_iter)
        except Exception:
            logger.exception("apply_named_tensors_in_place: model.load_weights failed")
            raise

        if sync:
            torch.cuda.synchronize()
        # The per-model loader logs its own counts; we return a coarse
        # tally for compatibility with callers reading the stats.
        n = len(named_tensors)
        stats = {
            "loaded": n,  # delegated; per-model loader reports detail
            "skipped_unknown": 0,
            "used_loader_cb": n,
            "used_direct_copy": 0,
            "via_model_load_weights": 1,
        }
        logger.info("apply_named_tensors_in_place: %s", stats)
        return stats

    # No model.load_weights — direct fallback (test models, simple stubs).
    logger.warning(
        "apply_named_tensors_in_place: model %s has no load_weights(); "
        "using direct param lookup. Fused weights (qkv_proj, gate_up_proj) "
        "will be skipped silently.",
        type(model).__name__,
    )
    return _apply_direct(model, named_tensors, sync=sync)


def _apply_direct(
    model: torch.nn.Module,
    named_tensors: dict[str, torch.Tensor],
    *,
    sync: bool,
) -> dict[str, int]:
    """Pre-refactor direct-lookup path. Kept for fallback only — does not
    handle packed-modules-mapping; use ``model.load_weights`` instead."""
    n_loaded = n_skipped = n_loader = n_direct = 0
    for name, full in named_tensors.items():
        try:
            param = model.get_parameter(name)
        except AttributeError:
            n_skipped += 1
            continue
        loader = getattr(param, "weight_loader", None)
        if loader is not None:
            loader(param, full)
            n_loader += 1
        else:
            param.data.copy_(full, non_blocking=True)
            n_direct += 1
        n_loaded += 1
    if sync:
        torch.cuda.synchronize()
    stats = {
        "loaded": n_loaded,
        "skipped_unknown": n_skipped,
        "used_loader_cb": n_loader,
        "used_direct_copy": n_direct,
        "via_model_load_weights": 0,
    }
    logger.info("apply_named_tensors_in_place (direct): %s", stats)
    return stats


class WeightUpdateEngine:
    """Coordinates RDMA weight pulls, local weight storage, and model apply."""

    def __init__(self, model: torch.nn.Module, weight_context: WeightContext):
        self.model = model
        self.weight_context = weight_context

    def apply_named_tensors(
        self, named_tensors: dict[str, torch.Tensor], *, sync: bool = True
    ) -> dict[str, int]:
        """Hot-load HF-named full tensors into the live model on this rank."""
        return apply_named_tensors_in_place(self.model, named_tensors, sync=sync)

    def apply_context(self, *, sync: bool = True) -> dict[str, int]:
        return self.apply_named_tensors(self.weight_context.named_tensors(), sync=sync)

    def pull_into_context(
        self, manifest_blob: bytes, train_alias: str
    ) -> dict[str, Any]:
        manifest = pickle.loads(manifest_blob)
        t0 = time.monotonic()
        self.weight_context.pull_named_tensors_via_rdma(train_alias, manifest)
        pull_s = time.monotonic() - t0
        return {
            "version": getattr(manifest, "version", None),
            "n_tensors": len(getattr(manifest, "entries", [])),
            "pull_s": pull_s,
        }

    def pull_and_apply(self, manifest_blob: bytes, train_alias: str) -> dict[str, Any]:
        """RDMA-pull a weight manifest from train side and hot-load it."""
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
