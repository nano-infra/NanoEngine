from typing import Any, Dict, List, Tuple

import ray

from nanodeploy.config import Config
from nanodeploy.engine.sequence import Sequence
from nanodeploy.worker.model_runner import ModelRunner


class RayExecutor:
    """Ray executor. Only support DP+EP+SP Mode"""

    def __init__(self, config: Config) -> None:

        self.config = config

        ray.init(address=config.ray_address, ignore_reinit_error=True)

        self.workers = []
        assert config.attn_world_size == config.ffn_world_size
        for global_rank in range(config.attn_world_size):
            self.workers.append(ModelRunner.remote(config, global_rank, None))

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
        tp_size = self.config.attention_tp
        dp_seqs = [num for num in dp_seqs for _ in range(tp_size)]
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
        tp_size = self.config.attention_tp
        sp_size = self.config.attention_sp
        dp_seqs = [seq for seq in dp_seqs for _ in range(tp_size * sp_size)]
        return ray.get(
            [
                getattr(worker, "run").remote(seqs, is_prefill)
                for seqs, worker in zip(dp_seqs, self.workers)
            ],
            timeout=timeout,
        )

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
