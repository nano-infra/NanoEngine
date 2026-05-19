from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PeerAgentContext:
    """Owns the worker PeerAgent lifecycle and transport settings."""

    agent: Any
    alias: str
    server_url: str
    device: str
    ib_port: int = 1
    qp_num: int = 1
    connected_peers: set[str] = field(default_factory=set)

    @classmethod
    def start_peer_agent(
        cls,
        *,
        nanoctrl_address: str | None,
        alias: str | None,
        device: str | None,
        scope: str | None = None,
        qp_num: int | None = None,
    ) -> "PeerAgentContext | None":
        """Start a DLSlime PeerAgent and return its public context handle."""
        if nanoctrl_address is None or alias is None:
            return None

        import dlslime

        start_peer_agent_fn = getattr(dlslime, "start_peer_agent", None)
        if not callable(start_peer_agent_fn):
            return None

        server_url = nanoctrl_address
        if not server_url.startswith("http://") and not server_url.startswith(
            "https://"
        ):
            server_url = f"http://{server_url}"

        if device is None:
            available_nics = dlslime.available_nic()
            if not available_nics:
                raise RuntimeError("No available NICs found")
            device = available_nics[0]

        agent = start_peer_agent_fn(
            nanoctrl_url=server_url,
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
        )

    @classmethod
    def start_for_cache_context(
        cls,
        cache_context: Any,
        *,
        rank: int,
    ) -> "PeerAgentContext | None":
        """Start a PeerAgent using transport settings from CacheContext."""
        return cls.start_peer_agent(
            nanoctrl_address=cache_context.nanoctrl_address,
            alias=(
                f"{cache_context.engine_id}:{rank}"
                if cache_context.engine_id is not None
                else None
            ),
            device=cache_context.selected_nic,
            scope=cache_context.nanoctrl_scope,
        )

    def is_connected(self, peer_alias: str) -> bool:
        """Return whether this PeerAgent already connected to ``peer_alias``."""
        return peer_alias in self.connected_peers

    def ensure_connected(self, peer_alias: str, *, timeout: float = 30) -> None:
        """Ensure the local PeerAgent is connected to ``peer_alias``."""
        if self.is_connected(peer_alias):
            return

        conn = self.agent.connect_to(
            peer_alias,
            ib_port=self.ib_port,
            qp_num=self.qp_num,
        )
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

        pending_conns = [
            self.agent.connect_to(
                peer,
                ib_port=self.ib_port,
                qp_num=self.qp_num,
            )
            for peer in new_peers
        ]
        for peer, conn in zip(new_peers, pending_conns, strict=True):
            if conn.wait(timeout=timeout) is False:
                raise RuntimeError(f"Timed out waiting for connection to {peer}")
        self.connected_peers.update(new_peers)
        return new_peers

    def unregister_memory_region(self, mr_name: str) -> None:
        """Unregister a local memory region from the owned PeerAgent."""
        self.agent.unregister_memory_region(mr_name)
