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
        for global_rank in range(config.world_size):
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

    def run(
        self, dp_seqs: List[List[Sequence]], is_prefill: bool, timeout: float = None
    ) -> list[int]:
        tp_size = self.config.attention_tp
        dp_seqs = [num for num in dp_seqs for _ in range(tp_size)]
        return ray.get(
            [
                getattr(worker, "run").remote(seqs, is_prefill)
                for seqs, worker in zip(dp_seqs, self.workers)
            ],
            timeout=timeout,
        )

    def num_kvcache_blocks(self):
        return self.collective_rpc("num_kvcache_blocks")

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
