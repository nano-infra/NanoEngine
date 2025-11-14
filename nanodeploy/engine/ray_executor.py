from typing import Any, Dict, List, Tuple

import ray
from ray.util.placement_group import placement_group, remove_placement_group

from nanodeploy.config import Config
from nanodeploy.engine.sequence import Sequence
from nanodeploy.logging import get_logger
from nanodeploy.worker.model_runner import ModelRunner


logger = get_logger()


def get_nodes_with_head_first():
    """
    node list
    """
    nodes = ray.nodes()

    # 自定义排序函数：头节点在前，其他节点按原顺序排列
    def sort_key(node):
        # 头节点返回 0，其他节点返回 1，确保头节点排在前面
        return 0 if "node:__internal_head__" in node.get("Resources", {}) else 1

    # 排序节点列表
    sorted_nodes = sorted(nodes, key=sort_key)
    return sorted_nodes


class RayExecutor:
    """Ray executor. Only support DP+EP+SP Mode"""

    def __init__(self, config: Config) -> None:
        self.config = config

        # 1. 初始化 Ray 连接
        ray.init(address=config.ray_address, ignore_reinit_error=True)

        self.workers = []
        self.placement_groups = []
        assert config.attn_world_size == config.ffn_world_size

        # 2. 获取所有节点的 NodeID
        nodes = get_nodes_with_head_first()
        node_ids = [node["NodeID"] for node in nodes]
        print(f"find nodes (NodeIDs): {node_ids}")

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
                bundles=[
                    {"CPU": 0.1, "GPU": 1.0} for _ in range(8)
                ],  # <-- 修改这里：使用最小化资源请求
                strategy="STRICT_PACK",
                name=f"pg-node-{node_idx}",
                _soft_target_node_id=target_node_id,
            )
            logger.info(target_node_id)

            ray.get(pg.ready())
            self.placement_groups.append(pg)

            start_rank = node_idx * workers_per_node
            end_rank = min(start_rank + workers_per_node, config.attn_world_size)

            for rank in range(start_rank, end_rank):
                worker = ModelRunner.options(placement_group=pg).remote(
                    config, rank, None
                )
                self.workers.append(worker)

        logger.info("\nAll workers scheduled successfully.")

    def __del__(self):
        if hasattr(self, "placement_groups") and self.placement_groups:
            for pg in self.placement_groups:
                try:
                    remove_placement_group(pg)
                except Exception as e:
                    logger.error(f"Warning: Failed to remove Placement Group: {e}")

    def collective_rpc(
        self,
        method: str,
        args: Tuple[Any] = None,
        kwargs: Dict[str, Any] = None,
        timeout: float = None,
    ):
        """Collective rpc."""
        if args is None:
            args = list()
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
        self, dp_seqs: List[List[Sequence]], timeout: float | None = None
    ) -> list[int]:
        return ray.get(
            [
                getattr(worker, "migrate").remote(seqs)
                for seqs, worker in zip(dp_seqs, self.workers)
            ],
            timeout=timeout,
        )

    def run(
        self, dp_seqs: List[List[Sequence]], is_prefill: bool, timeout: float = None
    ) -> list[int]:
        return ray.get(
            [
                getattr(worker, "run").remote(seqs, is_prefill)
                for seqs, worker in zip(dp_seqs, self.workers)
            ],
            timeout=timeout,
        )

    def update_kvcache_blocks(self):
        num_cache_blocks = min(self.collective_rpc("num_kvcache_blocks"))
        logger.info(f"Set {num_cache_blocks=}")
        self.collective_rpc("allocate_kvcache", (num_cache_blocks,))
        return num_cache_blocks

    def p2p_init(self, remote_name: str, num_kv_blocks: int, remote_world_size: int):
        return self.collective_rpc(
            "p2p_init", (remote_name, num_kv_blocks, remote_world_size)
        )

    def p2p_connect(self, remote_name: str, remote_endpoint_infos: list[list[dict]]):
        return self.collective_rpc("p2p_connect", (remote_name, remote_endpoint_infos))

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
