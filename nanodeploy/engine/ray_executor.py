import threading
import time
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


def _clean_and_parse_address(address: str) -> str:
    """
    清理并解析地址，正确处理 'ip:port' 格式。
    """
    # 如果地址包含 ':' 且不以 'http://' 或 'https://' 开头，我们认为它是 'ip:port' 格式
    if ":" in address and not address.startswith(("http://", "https://")):
        # 为其添加一个默认的 'http://' 前缀，使其成为一个标准 URL
        address = f"http://{address}"

    parsed_url = urlparse(address)

    # 如果解析后的 hostname 存在，则返回它
    if parsed_url.hostname:
        return parsed_url.hostname

    # 如果解析失败（例如，输入是一个纯 IP 或主机名），则返回原始地址
    return address


def get_available_nodes_with_master_first(master_address: str):
    """
    Retrieves a list of Ray nodes, sorting them so that the specified master node comes first.
    Excludes nodes that have any ALIVE Placement Groups.

    Args:
        master_address: The address of the master node.

    Returns:
        A list of available Ray node dictionaries, sorted with the master node first.
    """
    all_nodes = ray.nodes()
    if not all_nodes:
        logger.warning("No nodes found in the Ray cluster.")
        return []

    # --------------------------
    # Step 1: Clean and resolve the master address
    # --------------------------
    cleaned_host = _clean_and_parse_address(master_address)
    if cleaned_host in {"localhost", "127.0.0.1"}:
        if not ray.is_initialized():
            raise RuntimeError("Ray must be initialized to resolve 'localhost'")
        resolved_master_ip = ray.util.get_node_ip_address()
    else:
        resolved_master_ip = cleaned_host

    # --------------------------
    # Step 2: Get ALIVE Placement Groups and their nodes
    # --------------------------
    existing_pgs = ray.util.placement_group_table()
    nodes_with_alive_pg = set()

    for _, pg_info in existing_pgs.items():
        pg_state = pg_info.get("state", "")

        # Only consider ALIVE PGs
        if pg_state != "REMOVED":
            # A PG's bundles are spread across nodes. We need all nodes hosting its bundles.
            bundles_to_node_id = pg_info.get("bundles_to_node_id", {})
            for _, node_id in bundles_to_node_id.items():
                if node_id:
                    nodes_with_alive_pg.add(node_id)

    logger.info(f"Node IDs with ALIVE PGs: {nodes_with_alive_pg}")

    # --------------------------
    # Step 3: Filter available nodes
    # --------------------------
    available_nodes = [
        node for node in all_nodes if node["NodeID"] not in nodes_with_alive_pg
    ]

    # --------------------------
    # Step 4: Sort
    # --------------------------
    def sort_key(node):
        node_ip = node.get("NodeManagerAddress")
        logger.info(f"{node_ip=}, {resolved_master_ip=}")
        return 0 if node_ip == resolved_master_ip else 1

    sorted_available_nodes = sorted(available_nodes, key=sort_key)

    logger.info(f"Found {len(sorted_available_nodes)} available nodes.")

    assert sorted_available_nodes, "No available node resources"
    assert (
        sorted_available_nodes[0].get("NodeManagerAddress") == cleaned_host
    ), "master address is occupied or it is not mounted by ray."

    return sorted_available_nodes


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

        self.endpoint = RPCServerEndpoint(
            4*32_000_000, 
            self.config.attn_world_size,
            self.config.attention_sp,
            self.config.attention_tp,
            self.config.optimize_decode_block_table
        )

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
        self,
        dp_seqs: List[List[Sequence]],
        is_prefill: bool,
        timeout: float | None = None,
    ) -> list[list[list[int]]]:
        # start = time.perf_counter()
        send_timestamp = time.time()
        if self.config.use_dlslime_rpc:
            # When using dlslime RPC, sequences are delivered via the endpoint.
            ray_futures = [
                getattr(worker, "run").remote([], is_prefill, True, send_timestamp)
                for _, worker in zip(dp_seqs, self.workers)
            ]
            self.endpoint.send_seqs(dp_seqs, is_prefill)
            # trans_type = "DLSlime"
        else:
            # When not using dlslime RPC, pass sequences directly to workers.
            ray_futures = [
                getattr(worker, "run").remote(seqs, is_prefill, False, send_timestamp)
                for seqs, worker in zip(dp_seqs, self.workers)
            ]
            # trans_type = "Ray"

        # duration = (time.perf_counter() - start) * 1000
        # logger.info(f"[METRIC] Use DLSlime: {self.config.use_dlslime_rpc}, Duration: {duration:.4f} ms")

        results = ray.get(
            ray_futures,
            timeout=timeout,
        )
        recv_timestamp = time.time()

        token_ids_list = []
        worker_end_times = []
        for res in results:
            if isinstance(res, tuple) and len(res) == 2:
                token_ids_list.append(res[0])
                worker_end_times.append(res[1])
            else:
                token_ids_list.append(res)
        
        if worker_end_times:
            # Output Transfer Latency = Driver Recv Time - Max Worker Finish Time
            output_transfer_latency = (recv_timestamp - max(worker_end_times)) * 1000
            logger.info(f"[METRIC] Output Transfer Latency: {output_transfer_latency:.4f} ms")

        return token_ids_list

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
