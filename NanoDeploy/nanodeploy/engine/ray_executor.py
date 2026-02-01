import threading
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import ray
from ray.util.placement_group import placement_group, remove_placement_group

from nanodeploy.config import Config
from nanodeploy.endpoint.rpc_endpoint import RPCServerEndpoint
from nanodeploy.engine.sequence import Sequence
from nanodeploy.logging import get_logger
from nanodeploy.worker.model_runner import ModelRunner

logger = get_logger()


from nanodeploy.engine.ray_utils import get_available_nodes_with_master_first


class RayExecutor:
    """Ray executor. Only support DP+EP+SP Mode"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.lock = threading.Lock()

        # 1. 初始化 Ray 连接
        with self.lock:
            ray.init(address=config.ray_address, ignore_reinit_error=True)

        self.workers = []
        self.placement_groups = []
        assert config.attn_world_size == config.ffn_world_size

        # 2. 获取所有节点的 NodeID
        nodes = get_available_nodes_with_master_first(config.master_address)
        node_ids = [node["NodeID"] for node in nodes]
        logger.debug(f"find nodes (NodeIDs): {node_ids}")

        # 3. 定义每个节点上要运行的 worker 数量
        workers_per_node = 8

        # 4. 计算需要多少个节点
        num_nodes_needed = (
            config.attn_world_size + workers_per_node - 1
        ) // workers_per_node
        if num_nodes_needed > len(node_ids):
            raise ValueError(
                f"insufficient resources, {num_nodes_needed} on demand，but only find {len(node_ids)} nodes"
            )

        # 5. 为每个目标节点创建 Placement Group，并调度相应的 workers
        for node_idx in range(num_nodes_needed):
            target_node_id = node_ids[node_idx]
            logger.info(f"--- scheduling node: {target_node_id} ---")

            pg = placement_group(
                bundles=[{"CPU": 0.1, "GPU": 1.0} for _ in range(8)],
                strategy="STRICT_PACK",
                name=f"pg-node-{node_ids[node_idx]}",
                _soft_target_node_id=target_node_id,
            )

            ray.get(pg.ready())

            self.placement_groups.append(pg)

            start_rank = node_idx * workers_per_node
            end_rank = min(start_rank + workers_per_node, config.attn_world_size)

            for rank in range(start_rank, end_rank):
                worker = ModelRunner.options(placement_group=pg).remote(config, rank)
                self.workers.append(worker)

        self.endpoint = RPCServerEndpoint(32_000_000, self.config.attn_world_size)

        logger.info("All workers scheduled successfully.")

    def __del__(self):
        if hasattr(self, "workers") and self.workers:
            logger.info(f"Terminating {len(self.workers)} workers...")
            for worker in self.workers:
                try:
                    ray.kill(worker)
                    logger.debug(f"Worker {worker} terminated successfully.")
                except Exception as e:
                    logger.warning(f"Failed to terminate worker {worker}: {e}")
            del self.workers

        if hasattr(self, "placement_groups") and self.placement_groups:
            for pg in self.placement_groups:
                try:
                    remove_placement_group(pg)
                except Exception as e:
                    logger.error(f"Warning: Failed to remove Placement Group: {e}")

        logger.debug("Ray Executor deconstructed")

    def collective_rpc(
        self,
        method: str,
        args: tuple | None = None,
        kwargs: dict | None = None,
        timeout: float | None = None,
    ):
        """Collective rpc."""
        if args is None:
            args = tuple()
        if kwargs is None:
            kwargs = dict()

        return ray.get(
            [
                getattr(worker, method).remote(*args, **kwargs)
                for worker in self.workers
            ],
            timeout=timeout,
        )

    def migrate(
        self,
        dp_seqs: List[List[Sequence]],
        peer_endpoints: dict[str, list[str]] = None,
        timeout: float | None = None,
    ) -> list[int]:
        return ray.get(
            [
                getattr(worker, "migrate").remote(seqs, peer_endpoints)
                for seqs, worker in zip(dp_seqs, self.workers)
            ],
            timeout=timeout,
        )

    def run(
        self,
        dp_seqs: List[List[Sequence]],
        is_prefill: bool,
        timeout: float | None = None,
    ) -> list[list[list[int]]]:

        if self.config.use_dlslime_rpc:
            # When using dlslime RPC, sequences are delivered via the endpoint.
            ray_futures = [
                getattr(worker, "run").remote([], is_prefill, True)
                for _, worker in zip(dp_seqs, self.workers)
            ]
            self.endpoint.send_seqs(dp_seqs, is_prefill)
        else:
            # When not using dlslime RPC, pass sequences directly to workers.
            ray_futures = [
                getattr(worker, "run").remote(seqs, is_prefill, False)
                for seqs, worker in zip(dp_seqs, self.workers)
            ]
        return ray.get(
            ray_futures,
            timeout=timeout,
        )

    def init_rpc_endpoint(self):
        info = self.endpoint.init_server_endpoint()
        client_info = self.collective_rpc("init_rpc_endpoint", (info,))
        self.endpoint.connect(client_info)
        logger.info("Server endpoint initialized")
        return 0

    def update_kvcache_blocks(self):
        num_cache_blocks = min(self.collective_rpc("num_kvcache_blocks"))
        logger.info(f"Set {num_cache_blocks=}")
        self.collective_rpc("allocate_kvcache", (num_cache_blocks,))
        return num_cache_blocks

    def p2p_init(self, remote_name: str, num_kv_blocks: int, remote_world_size: int):
        return self.collective_rpc(
            "p2p_init", (remote_name, num_kv_blocks, remote_world_size)
        )

    def p2p_connect(self, remote_name: str, remote_endpoint_infos: list[bytes]):
        return self.collective_rpc("p2p_connect", (remote_name, remote_endpoint_infos))

    def get_peer_agent_addrs(self) -> list[str]:
        """Get peer agent addresses from all workers."""
        return self.collective_rpc("get_peer_agent_addr")

    def p2p_disconnect(self, remote_name: str):
        return self.collective_rpc("p2p_disconnect", (remote_name,))

    def gather_free_mem(self):
        """Get free memory."""
        return self.collective_rpc("get_free_mem")

    def get_cache_block_size(self, block_size, world_size):
        """Get cache block size."""
        return self.collective_rpc("get_cache_block_size", (block_size, world_size))

    def allocate_kvcache(self, num_block_per_rank):
        """Allocate kv cache."""
        return self.collective_rpc("allocate_gpu_cache", args=(num_block_per_rank,))

    def init_cudagraph_buffer(self):
        """Initialize cuda graph buffer."""
        return self.collective_rpc("init_cudagraph_buffer")

    def capture_cudagraph(self):
        """Capture cuda graph."""
        return self.collective_rpc("capture_cudagraph")
