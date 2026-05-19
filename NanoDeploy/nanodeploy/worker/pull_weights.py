"""Direct RDMA weight pull for ModelRunner workers.

Replaces the slow path (rollout-driver pulls 8 GB to CPU, then Ray-RPCs
the full dict to each of N workers). Instead each worker uses its *own*
``PeerAgent`` (started through ``PeerAgentContext.start_peer_agent``)
to pull the manifest in parallel from the train side.

Speedup model:
    OLD: pull_to_driver + N * ray_serialize(8 GB)   ~ 2.9s + 4 * (10–20s)
    NEW: max_per_worker(pull_8GB)                   ~ 4–6s on a single NIC
                                                    ~ 1–2s if NICs span peers

For TP-sharded params each worker still pulls the FULL HF tensor and
slices via the parameter's ``weight_loader``. A future optimization
slices server-side and registers per-rank MRs to halve the data each
worker actually moves.
"""

from __future__ import annotations

import logging
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


def pull_named_tensors_via_rdma(
    peer_context: PeerAgentContext,
    train_alias: str,
    manifest,
) -> tuple[dict[str, torch.Tensor], list]:
    """Pull every entry of ``manifest`` from ``train_alias`` into local CPU
    receive buffers via a single batched ``endpoint.read``.

    Returns ``(named_tensors, registered_mr_names)`` — the caller must
    keep ``named_tensors`` alive until ``apply_named_tensors_in_place``
    has copied them, then can call ``unregister_memory_region`` on each
    name in ``registered_mr_names`` to free them.
    """
    peer_context.ensure_connected(train_alias, timeout=60.0)
    peer_agent = peer_context.agent
    endpoint = peer_agent._get_endpoint(train_alias)
    conn = peer_agent._get_connection(train_alias)

    received: dict[str, torch.Tensor] = {}
    mr_names: list[str] = []
    assigns: list[tuple[int, int, int, int, int]] = []

    for entry in manifest.entries:
        dtype = _STR_TO_DTYPE[entry.dtype]
        buf = torch.empty(
            tuple(entry.shape), dtype=dtype, device="cpu", pin_memory=True
        )
        received[entry.name] = buf
        peer_agent.register_memory_region(entry.mr_name, buf.data_ptr(), 0, entry.size)
        mr_names.append(entry.mr_name)
        local_handle = peer_agent.get_handle(entry.mr_name, resource_key=conn.local_key)
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
    return received, mr_names


def pull_and_apply_on_worker(
    model: torch.nn.Module,
    peer_context: PeerAgentContext,
    train_alias: str,
    manifest_blob: bytes,
) -> dict[str, Any]:
    """Compatibility wrapper around ``WeightUpdateEngine``."""
    from nanodeploy.context.weight import WeightContext
    from nanodeploy.worker.weight_update_engine import WeightUpdateEngine

    weight_context = WeightContext(peer_context)
    return WeightUpdateEngine(model, weight_context).pull_and_apply(
        manifest_blob, train_alias
    )
