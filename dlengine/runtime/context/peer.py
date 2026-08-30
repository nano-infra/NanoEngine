from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from dlengine.runtime.context import BaseContext


def normalize_peer_placements(
    placements: list[dict | None], world_size: int
) -> list[dict]:
    """Validate an engine's placement publication or preserve RDMA mode."""
    if len(placements) != world_size:
        raise RuntimeError(
            f"Engine expected {world_size} worker placements, got {len(placements)}"
        )
    present = [placement for placement in placements if placement is not None]
    if not present:
        return []
    if len(present) != world_size:
        raise RuntimeError(
            "Engine has partial Fabric placement metadata; every worker must "
            "publish placement or every worker must use the RDMA path"
        )
    ranks = sorted(int(placement["rank"]) for placement in present)
    if ranks != list(range(world_size)):
        raise RuntimeError(f"Engine placement ranks are invalid: {ranks}")
    domains = {str(placement["fabric_domain_id"]) for placement in present}
    if len(domains) != 1:
        raise RuntimeError(
            f"Engine workers span incompatible Fabric domains: {sorted(domains)}"
        )
    return sorted(present, key=lambda placement: int(placement["rank"]))


@dataclass
class PeerContext(BaseContext):
    """Owns the worker PeerAgent lifecycle and transport settings."""

    agent: Any
    alias: str
    server_url: str
    device: str | None
    rank: int = 0
    memory_regions: dict[str, Any] = field(default_factory=dict)
    ib_port: int = 1
    qp_num: int = 1
    connected_peers: set[str] = field(default_factory=set)

    @classmethod
    def get_context_type(cls) -> str:
        return "peer"

    @classmethod
    def get_context_name(cls) -> str:
        return "PeerContext"

    @classmethod
    def start_peer_agent(
        cls,
        *,
        ctrl_address: str | None,
        alias: str | None,
        device: str | None,
        scope: str | None = None,
        qp_num: int | None = None,
    ) -> "PeerContext | None":
        """Start a DLSlime PeerAgent and return its public context handle."""
        if ctrl_address is None or alias is None:
            return None

        import dlslime

        start_peer_agent_fn = getattr(dlslime, "start_peer_agent", None)
        if not callable(start_peer_agent_fn):
            return None

        server_url = ctrl_address
        if not server_url.startswith("http://") and not server_url.startswith(
            "https://"
        ):
            server_url = f"http://{server_url}"

        agent = start_peer_agent_fn(
            ctrl_url=server_url,
            alias=alias,
            device=device,
            scope=scope,
        )
        return cls(
            agent=agent,
            alias=alias,
            server_url=server_url,
            device=device,
            ib_port=1,
            qp_num=int(os.environ.get("SLIME_QP_NUM", 1) if qp_num is None else qp_num),
            rank=0,
        )

    @classmethod
    def start_for_cache_context(
        cls,
        cache_context: Any,
        *,
        rank: int,
    ) -> "PeerContext | None":
        """Start a PeerAgent using transport settings from CacheContext."""
        alias = (
            f"{cache_context.engine_id}:{rank}"
            if cache_context.engine_id is not None
            else None
        )
        context = cls.start_peer_agent(
            ctrl_address=cache_context.ctrl_address,
            alias=alias,
            device=None,
            scope=cache_context.ctrl_scope,
        )
        if context is not None:
            context.rank = rank
        return context

    def supports_cuda_fabric(self) -> bool:
        """Return whether the local worker published usable CUDA Fabric topology."""
        if not callable(getattr(self.agent, "allocate_memory_region", None)):
            return False
        resource = self.agent.get_resource(self.alias) or {}
        cuda_caps = (resource.get("runtime_capabilities") or {}).get("cuda") or {}
        imex_channels = (cuda_caps.get("imex") or {}).get("channel_ids") or []
        accelerators = resource.get("accelerators") or []
        ready = [
            accelerator
            for accelerator in accelerators
            if (accelerator.get("mnnvl") or {}).get("membership_ready")
        ]
        return len(ready) == 1 and bool(imex_channels)

    def local_placement(self) -> dict[str, Any] | None:
        """Return normalized placement metadata for the visible Fabric GPU."""
        if not self.supports_cuda_fabric():
            return None
        resource = self.agent.get_resource(self.alias) or {}
        cuda_caps = (resource.get("runtime_capabilities") or {}).get("cuda") or {}
        accelerators = [
            accelerator
            for accelerator in (resource.get("accelerators") or [])
            if (accelerator.get("mnnvl") or {}).get("membership_ready")
        ]
        accelerator = accelerators[0]
        fabric = accelerator["mnnvl"]
        cluster_uuid = str(fabric["cluster_uuid"]).lower()
        clique_id = int(fabric["clique_id"])
        return {
            "rank": self.rank,
            "peer_agent_id": self.alias,
            "gpu_uuid": accelerator["uuid"],
            "cluster_uuid": cluster_uuid,
            "clique_id": clique_id,
            "fabric_domain_id": f"{cluster_uuid}:{clique_id}",
            "topology_epoch": int(resource.get("topology_epoch", 0)),
            "membership_ready": True,
            "imex_channel_ids": list(
                ((cuda_caps.get("imex") or {}).get("channel_ids") or [])
            ),
        }

    def allocate_tensor(
        self, name: str, shape: tuple[int, ...], dtype: Any, *, zero: bool = False
    ):
        """Create a PyTorch tensor backed by a PeerAgent-owned Fabric region."""
        import math

        import torch

        if name in self.memory_regions:
            raise ValueError(f"Fabric memory region {name!r} already exists")
        numel = math.prod(int(dim) for dim in shape)
        itemsize = torch.empty((), dtype=dtype).element_size()
        region = self.agent.allocate_memory_region(name, numel * itemsize)

        class _CudaBytes:
            def __init__(self, ptr: int, size: int) -> None:
                self.__cuda_array_interface__ = {
                    "shape": (size,),
                    "typestr": "|u1",
                    "data": (ptr, False),
                    "version": 3,
                    "strides": None,
                }

        owner = _CudaBytes(region.ptr, region.length)
        raw = torch.as_tensor(owner, device="cuda")
        tensor = raw.view(dtype).view(shape)
        if zero:
            tensor.zero_()
        # Retain every owner for at least as long as the tensor/cache context.
        self.memory_regions[name] = (region, owner, raw)
        return tensor

    def owns_memory_region(self, name: str) -> bool:
        return name in self.memory_regions

    def _connect_to(self, peer_alias: str):
        """Prefer automatic transport selection with old-DLSlime RDMA fallback."""
        try:
            return self.agent.connect_to(
                peer_alias,
                transport="auto",
                ib_port=self.ib_port,
                qp_num=self.qp_num,
            )
        except ValueError as error:
            if "unsupported transport" not in str(error):
                raise
            return self.agent.connect_to(
                peer_alias, ib_port=self.ib_port, qp_num=self.qp_num
            )

    def is_connected(self, peer_alias: str) -> bool:
        """Return whether this PeerAgent already connected to ``peer_alias``."""
        return peer_alias in self.connected_peers

    def ensure_connected(self, peer_alias: str, *, timeout: float = 30) -> None:
        """Ensure the local PeerAgent is connected to ``peer_alias``."""
        if self.is_connected(peer_alias):
            return

        conn = self._connect_to(peer_alias)
        if conn.wait(timeout=timeout) is False:
            raise RuntimeError(f"Timed out waiting for connection to {peer_alias}")
        self.connected_peers.add(peer_alias)

    def ensure_many_connected(
        self, peer_aliases: list[str], *, timeout: float = 30
    ) -> list[str]:
        """Connect to missing peers and return the newly connected aliases."""
        new_peers = [peer for peer in peer_aliases if not self.is_connected(peer)]
        if not new_peers:
            return []

        pending_conns = [self._connect_to(peer) for peer in new_peers]
        newly_connected = []
        for peer, conn in zip(new_peers, pending_conns, strict=True):
            if conn.wait(timeout=timeout) is False:
                raise RuntimeError(f"Timed out waiting for connection to {peer}")
            self.connected_peers.add(peer)
            newly_connected.append(peer)
        return newly_connected

    def unregister_memory_region(self, mr_name: str) -> None:
        """Unregister a local memory region from the owned PeerAgent."""
        self.agent.unregister_memory_region(mr_name)

    def clear_context(self) -> None:
        self.connected_peers.clear()
        regions = list(self.memory_regions.values())
        self.memory_regions.clear()
        for region, _owner, _raw in regions:
            region.close()

    def reset_context(self) -> None:
        self.clear_context()


PeerAgentContext = PeerContext


def _select_cache_peer_device() -> str | None:
    from dlengine.runtime.disagg.p2p import select_peer_device

    return select_peer_device()


__all__ = ["PeerAgentContext", "PeerContext", "normalize_peer_placements"]
