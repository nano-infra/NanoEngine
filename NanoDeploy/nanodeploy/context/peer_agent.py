from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PeerAgentContext:
    """Owns the worker PeerAgent lifecycle and transport settings."""

    agent: Any
    alias: str
    server_url: str
    device: str
    ib_port: int = 1
    qp_num: int = 1

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
