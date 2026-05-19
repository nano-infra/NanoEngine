from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from nanodeploy.context.peer_agent import PeerAgentContext

logger = logging.getLogger("nanodeploy")


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
