import threading
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import ray
from ray.util.placement_group import placement_group, remove_placement_group

from nanodeploy._cpp import Sequence, serialize_migrate_batch, serialize_run_batch
from nanodeploy.config import Config
from nanodeploy.logging import get_logger
from nanodeploy.worker.model_runner import ModelRunner

logger = get_logger()


def _serialize_run(seqs: list[Sequence], is_prefill: bool) -> bytes:
    """Serialize sequences into lean RunBatchInput bytes."""
    return serialize_run_batch(seqs, is_prefill)


def _serialize_migrate(seqs: list[Sequence]) -> bytes:
    """Serialize sequences into lean MigrateBatchInput bytes."""
    return serialize_migrate_batch(seqs)


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

        # Check if running under NanoOps orchestration
        import os

        nanoops_pg_id = os.getenv("NANOOPS_PLACEMENT_GROUP_ID")

        if nanoops_pg_id:
            # NanoOps orchestration mode: use pre-created placement group
            logger.info(
                f"Using pre-created placement group from NanoOps: {nanoops_pg_id}"
            )
            self._init_with_existing_pg(nanoops_pg_id)
            self.external_pg = True  # Mark as externally managed
        else:
            # Manual launch mode: create placement groups as before
            logger.info("Creating placement groups (manual launch mode)")
            self._init_with_new_pg()
            self.external_pg = False

        logger.info("All workers scheduled successfully.")

    def _init_with_existing_pg(self, pg_id_hex: str):
        """Initialize workers using pre-created placement group from NanoOps.

        When running inside a Ray job with placement_group_id in runtime_env,
        all tasks are automatically scheduled to that placement group.
        We don't need to explicitly pass the PG to workers.

        Args:
            pg_id_hex: Placement group ID in hex format
        """
        logger.info(
            f"Scheduling {self.config.attn_world_size} workers using Ray job's placement group"
        )

        # --- Phase 1: create all workers with deferred dist init ----------
        # The master_address from config points to the Ray head node, but
        # workers may be on a different node.  We create them with
        # defer_dist_init=True so they skip dist.init_process_group().
        for rank in range(self.config.attn_world_size):
            worker = ModelRunner.remote(self.config, rank, defer_dist_init=True)
            self.workers.append(worker)

        # --- Phase 2: probe worker[0] for actual node IP + free port ------
        worker_ip, free_port = ray.get(self.workers[0].get_node_info.remote())
        master_address = f"{worker_ip}:{free_port}"
        logger.info(
            f"Probed worker node: master_address = {master_address} "
            f"(was {self.config.master_address})"
        )
        self.config.master_address = master_address

        # --- Phase 3: trigger dist init on ALL workers simultaneously -----
        # dist.init_process_group is a collective call; all ranks must
        # enter it together.
        init_futures = [w.init_dist.remote(master_address) for w in self.workers]
        ray.get(init_futures)
        logger.info("All workers completed distributed init.")

        # No placement group object to store - managed by Ray job runtime
        self.placement_groups = []

    def _init_with_new_pg(self):
        """Initialize workers by creating new placement groups (existing logic)."""
        # 2. 获取所有节点的 NodeID
        nodes = get_available_nodes_with_master_first(self.config.master_address)
        node_ids = [node["NodeID"] for node in nodes]
        logger.debug(f"find nodes (NodeIDs): {node_ids}")

        # 3. 定义每个节点上要运行的 worker 数量
        workers_per_node = 8

        # 4. 计算需要多少个节点
        num_nodes_needed = (
            self.config.attn_world_size + workers_per_node - 1
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
            end_rank = min(start_rank + workers_per_node, self.config.attn_world_size)

            for rank in range(start_rank, end_rank):
                worker = ModelRunner.options(placement_group=pg).remote(
                    self.config, rank
                )
                self.workers.append(worker)

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

        # Only remove placement groups if we created them (not externally managed)
        if hasattr(self, "placement_groups") and self.placement_groups:
            if hasattr(self, "external_pg") and self.external_pg:
                logger.info(
                    "Skipping placement group removal (externally managed by NanoOps)"
                )
            else:
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
        timeout: float | None = None,
    ) -> list[int]:
        # Serialize into lean MigrateBatchInput bytes, send bytes instead of Sequence objects
        batch_bytes = [_serialize_migrate(seqs) for seqs in dp_seqs]
        return ray.get(
            [
                getattr(worker, "migrate_from_bytes").remote(b)
                for b, worker in zip(batch_bytes, self.workers)
            ],
            timeout=timeout,
        )

    def run(
        self,
        dp_seqs: List[List[Sequence]],
        is_prefill: bool,
        timeout: float | None = None,
    ) -> list[list[list[int]]]:
        # Serialize into lean RunBatchInput bytes, send bytes instead of Sequence objects
        batch_bytes = [_serialize_run(seqs, is_prefill) for seqs in dp_seqs]
        ray_futures = [
            getattr(worker, "run_from_bytes").remote(b, is_prefill)
            for b, worker in zip(batch_bytes, self.workers)
        ]
        return ray.get(
            ray_futures,
            timeout=timeout,
        )

    def update_kvcache_blocks(self):
        num_cache_blocks = min(self.collective_rpc("num_kvcache_blocks"))
        logger.info(f"Set {num_cache_blocks=}")
        self.collective_rpc("allocate_kvcache", (num_cache_blocks,))
        return num_cache_blocks

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
