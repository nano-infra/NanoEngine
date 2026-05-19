from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PeerAgentContext:
    """Public PeerAgent transport handle shared by cache and weight-sync code."""

    agent: Any
    alias: str
    ib_port: int = 1
    qp_num: int = 1
