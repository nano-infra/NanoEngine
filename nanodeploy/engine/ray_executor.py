import threading
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import ray
from ray.util.placement_group import placement_group, remove_placement_group

from nanodeploy.config import Config

# 引入 C++ 扩展中的 Batch 类
from nanodeploy.engine._core import SequenceBatch
from nanodeploy.engine.sequence import Sequence
from nanodeploy.logging import get_logger
from nanodeploy.worker.model_runner import ModelRunner

logger = get_logger()


def _clean_and_parse_address(address: str) -> str:
    """清理并解析地址，正确处理 'ip:port' 格式。"""
    if ":" in address and not address.startswith(("http://", "https://")):
        address = f"http://{address}"
    parsed_url = urlparse(address)
    if parsed_url.hostname:
        return parsed_url.hostname
    return address


def get_available_nodes_with_master_first(master_address: str):
    """
    获取可用节点列表，Master 节点排在第一位。
    """
    all_nodes = ray.nodes()
    if not all_nodes:
        logger.warning("No nodes found in the Ray cluster.")
        return []

    cleaned_host = _clean_and_parse_address(master_address)
    if cleaned_host in {"localhost", "127.0.0.1"}:
        if not ray.is_initialized():
            raise RuntimeError("Ray must be initialized to resolve 'localhost'")
        resolved_master_ip = ray.util.get_node_ip_address()
    else:
        resolved_master_ip = cleaned_host

    existing_pgs = ray.util.placement_group_table()
    nodes_with_alive_pg = set()

    for _, pg_info in existing_pgs.items():
        pg_state = pg_info.get("state", "")
        if pg_state != "REMOVED":
            bundles_to_node_id = pg_info.get("bundles_to_node_id", {})
            for _, node_id in bundles_to_node_id.items():
                if node_id:
                    nodes_with_alive_pg.add(node_id)

    available_nodes = [
        node for node in all_nodes if node["NodeID"] not in nodes_with_alive_pg
    ]

    def sort_key(node):
        node_ip = node.get("NodeManagerAddress")
        return 0 if node_ip == resolved_master_ip else 1

    sorted_available_nodes = sorted(available_nodes, key=sort_key)
    assert sorted_available_nodes, "No available node resources"
    return sorted_available_nodes


class RayExecutor:
    """Ray executor. Only support DP+EP+SP Mode"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.lock = threading.Lock()

        with self.lock:
            ray.init(address=config.ray_address, ignore_reinit_error=True)

        self.workers = []
        self.placement_groups = []
        assert config.attn_world_size == config.ffn_world_size

        nodes = get_available_nodes_with_master_first(config.master_address)
        node_ids = [node["NodeID"] for node in nodes]
        print(f"find nodes (NodeIDs): {node_ids}")

        workers_per_node = 8
        num_nodes_needed = (
            config.attn_world_size + workers_per_node - 1
        ) // workers_per_node

        if num_nodes_needed > len(node_ids):
            raise ValueError(
                f"insufficient resources, {num_nodes_needed} on demand，but only find {len(node_ids)} nodes"
            )

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

        logger.info("All workers scheduled successfully.")

    def __del__(self):
        if hasattr(self, "workers") and self.workers:
            for worker in self.workers:
                try:
                    ray.kill(worker)
                except Exception:
                    pass
            del self.workers

        if hasattr(self, "placement_groups") and self.placement_groups:
            for pg in self.placement_groups:
                try:
                    remove_placement_group(pg)
                except Exception:
                    pass

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
        self, dp_seqs: List[List[Sequence]], timeout: float | None = None
    ) -> list[int]:
        # [优化] 使用 SequenceBatch 包装列表，实现零拷贝二进制传输
        # SequenceBatch 在 C++ 层实现了高效的 pickle 协议
        batched_args = [SequenceBatch(seqs) for seqs in dp_seqs]

        return ray.get(
            [
                getattr(worker, "migrate").remote(seqs_batch)
                for seqs_batch, worker in zip(batched_args, self.workers)
            ],
            timeout=timeout,
        )

    def run(
        self,
        dp_seqs: List[List[Sequence]],
        is_prefill: bool,
        timeout: float | None = None,
    ) -> tuple[list[list[list[int]]], float, float]:
        # 如果 is_prefill 为 True，则 is_decode 为 False (传输全量)
        # 如果 is_prefill 为 False，则 is_decode 为 True (只传最后一个 token)
        is_decode = not is_prefill

        # 使用 SequenceBatch 包装，大幅降低序列化开销
        # 对 Ray 来说，这是一个单一对象传输，而非数千个小对象
        batched_args = [SequenceBatch(seqs, is_decode) for seqs in dp_seqs]

        results = ray.get(
            [
                getattr(worker, "run").remote(seqs_batch, is_prefill)
                for seqs_batch, worker in zip(batched_args, self.workers)
            ],
            timeout=timeout,
        )

        token_ids = [res[0] for res in results]
        run_latencies = [res[1] for res in results]
        model_latencies = [res[2] for res in results]

        return token_ids, run_latencies, model_latencies

    # ... [下面的辅助方法保持不变] ...
    def update_kvcache_blocks(self):
        num_cache_blocks = min(self.collective_rpc("num_kvcache_blocks"))
        self.collective_rpc("allocate_kvcache", (num_cache_blocks,))
        return num_cache_blocks

    def p2p_init(self, remote_name: str, num_kv_blocks: int, remote_world_size: int):
        return self.collective_rpc(
            "p2p_init", (remote_name, num_kv_blocks, remote_world_size)
        )

    def p2p_connect(self, remote_name: str, remote_endpoint_infos: list[list[dict]]):
        return self.collective_rpc("p2p_connect", (remote_name, remote_endpoint_infos))

    def gather_free_mem(self):
        return self.collective_rpc("get_free_mem")

    def get_cache_block_size(self, block_size, world_size):
        return self.collective_rpc("get_cache_block_size", (block_size, world_size))

    def allocate_kvcache(self, num_block_per_rank):
        return self.collective_rpc("allocate_gpu_cache", args=(num_block_per_rank,))

    def init_cudagraph_buffer(self):
        return self.collective_rpc("init_cudagraph_buffer")

    def capture_cudagraph(self):
        return self.collective_rpc("capture_cudagraph")
