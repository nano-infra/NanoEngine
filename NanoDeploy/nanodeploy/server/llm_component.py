import json
from typing import List, Set, Tuple

import nanodeploy.fbs.EngineInfo as EngineInfo
from nanodeploy.config import Config
from nanodeploy.engine.llm_engine import LLMEngine
from nanodeploy.llm import LLM
from nanodeploy.logging import get_logger

logger = get_logger("nanodeploy")


class LLMComponent(LLM):
    def __init__(self, config: Config):
        super().__init__(config)

        # peer_engine_id -> dict(num_blocks, world_size, peer_addrs) for lazy migration
        self._peer_info: dict[str, dict] = {}
        self.active_p2p_links: Set[str] = set()

    def get_engine_info(self, status: str = "ready") -> str:
        """Get engine info as JSON string."""
        # Get peer_agent addresses from all workers
        peer_addrs = (
            self.executor.get_peer_agent_addrs()
            if hasattr(self.executor, "get_peer_agent_addrs")
            else []
        )

        engine_info = {
            "id": self.engine_id,
            "role": self.config.mode,
            "rank": 0,
            "world_size": self.config.attn_world_size,
            "num_blocks": self.config.num_kvcache_blocks,
            "host": self.config.host,
            "port": self.config.port,
            "status": status,
            "peer_addrs": peer_addrs,
        }

        return json.dumps(engine_info)

    def set_peer_info(self, remote_engine_info: str | bytes) -> None:
        """Store remote engine info including peer_addrs for lazy migration.

        This method parses the remote engine's info and stores it so that during
        migration, the endpoints can be embedded in BlockContext.endpoints for
        lazy P2P connection.

        Args:
            remote_engine_info: JSON string or FlatBuffers bytes (for backward compatibility)
        """
        # Try to parse as JSON first
        if isinstance(remote_engine_info, bytes):
            try:
                # Try JSON first
                remote_engine_info = remote_engine_info.decode("utf-8")
                info_dict = json.loads(remote_engine_info)
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Fall back to FlatBuffers for backward compatibility
                info = EngineInfo.EngineInfo.GetRootAsEngineInfo(remote_engine_info, 0)
                remote_engine_id = info.Id().decode("utf-8") if info.Id() else ""
                num_kv_blocks = info.NumBlocks()

                # Extract peer_addrs from EngineInfo
                peer_addrs = []
                for i in range(info.PeerAddrsLength()):
                    addr = info.PeerAddrs(i)
                    if addr:
                        peer_addrs.append(
                            addr.decode("utf-8") if isinstance(addr, bytes) else addr
                        )

                # Store in engine for use during migration
                self._peer_info[remote_engine_id] = {
                    "num_blocks": num_kv_blocks,
                    "world_size": info.WorldSize(),
                    "peer_addrs": peer_addrs,
                }
                logger.info(
                    f"Stored peer info for {remote_engine_id}: {len(peer_addrs)} addresses"
                )
                return
        else:
            info_dict = json.loads(remote_engine_info)

        # Parse JSON format
        remote_engine_id = info_dict.get("id", "")
        num_kv_blocks = info_dict.get("num_blocks", 0)
        world_size = info_dict.get("world_size", 1)
        peer_addrs = info_dict.get("peer_addrs", [])

        # Store in engine for use during migration
        self._peer_info[remote_engine_id] = {
            "num_blocks": num_kv_blocks,
            "world_size": world_size,
            "peer_addrs": peer_addrs,
        }
        logger.info(
            f"Stored peer info for {remote_engine_id}: {len(peer_addrs)} addresses"
        )

    def ensure_p2p_connected(self, peer_id: str) -> None:
        """Ensure P2P link to peer_id is established (lazy connect). Idempotent."""
        if peer_id in self.active_p2p_links:
            return
        if peer_id not in self._peer_info:
            logger.warning(
                f"ensure_p2p_connected: no peer info for {peer_id}, skipping"
            )
            return
        peer_info = self._peer_info[peer_id]
        if isinstance(peer_info, tuple):
            # Legacy format: (addrs, num_blocks)
            addrs, num_blocks = peer_info
        else:
            # New format: dict with peer_addrs
            addrs = peer_info.get("peer_addrs", [])
            num_blocks = peer_info.get("num_blocks", 0)
        try:
            self.executor.ensure_p2p_connected(peer_id, addrs, num_blocks)
            self.active_p2p_links.add(peer_id)
            logger.info(f"P2P link ensured to {peer_id}")
        except Exception as e:
            logger.error(f"ensure_p2p_connected failed for {peer_id}: {e}")
            raise
