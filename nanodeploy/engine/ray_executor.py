from typing import Any, Dict, List, Tuple

import ray

from nanodeploy.config import Config
from nanodeploy.worker.model_runner import ModelRunner


class RayExecutor:
    """Ray executor. Only support DP+EP+SP Mode"""

    def __init__(self, config: Config) -> None:

        ray.init(address=config.ray_address, ignore_reinit_error=True)

        self.workers = []
        for global_rank in range(config.world_size):
            self.workers.append(ModelRunner(config, global_rank, None))

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

    def step(self):
        raise NotImplementedError

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

    def forward_disable_sp(
        self,
        cache_seqlens_list: List[List[int]],
        block_table_list: List[List[List[int]]],
    ):
        """
        为每个worker分配列表中对应位置的参数并执行推理（禁用SP模式）
        Args:
            cache_seqlens_list: 每个worker的cache_seqlens列表（int类型）
            block_table_list: 每个worker的block_table列表
        Returns:
            所有worker的推理结果列表（与worker顺序一致）
        """
        worker_count = len(self.workers)
        param_lists = [cache_seqlens_list, block_table_list]
        param_names = ["cache_seqlens_list", "block_table_list"]

        for param_list, param_name in zip(param_lists, param_names):
            assert (
                len(param_list) == worker_count
            ), f"{param_name}长度({len(param_list)})与worker数量({worker_count})不匹配"

        futures = []
        for i, worker in enumerate(self.workers):
            futures.append(
                worker.run_model.remote(cache_seqlens_list[i], block_table_list[i])
            )

        return ray.get(futures)

    def forward_enable_sp(
        self,
        local_cnt_list: List[int],
        sp_cnt_list: List[int],
        group_sp_cnt_list: List[List[int]],
        cache_seqlens_list: List[List[int]],
        block_table_list: List[List[List[int]]],
    ):
        """
        为每个worker分配列表中对应位置的参数并执行推理（启用SP模式）
        Args:
            local_cnt_list: 每个worker的local_cnt列表
            sp_cnt_list: 每个worker的sp_cnt列表
            group_sp_cnt_list: 每个worker的group_sp_cnt列表
            cache_seqlens_list: 每个worker的cache_seqlens列表（int类型）
            block_table_list: 每个worker的block_table列表
        Returns:
            所有worker的推理结果列表（与worker顺序一致）
        """
        worker_count = len(self.workers)
        param_lists = [
            local_cnt_list,
            sp_cnt_list,
            group_sp_cnt_list,
            cache_seqlens_list,
            block_table_list,
        ]
        param_names = [
            "local_cnt_list",
            "sp_cnt_list",
            "group_sp_cnt_list",
            "cache_seqlens_list",
            "block_table_list",
        ]

        for param_list, param_name in zip(param_lists, param_names):
            assert (
                len(param_list) == worker_count
            ), f"{param_name}长度({len(param_list)})与worker数量({worker_count})不匹配"

        futures = []
        for i, worker in enumerate(self.workers):
            futures.append(
                worker.run_model.remote(
                    local_cnt_list[i],
                    sp_cnt_list[i],
                    group_sp_cnt_list[i],
                    cache_seqlens_list[i],
                    block_table_list[i],
                )
            )

        return ray.get(futures)

    def step_sp(
        self,
        local_cnt_list: List[int],
        sp_cnt_list: List[int],
        group_sp_cnt_list: List[List[int]],
        cache_seqlens_list: List[List[int]],
        block_table_list: List[List[List[int]]],
        decode_loopcount: int,
    ):
        """
        为每个worker分配列表中对应位置的参数并执行推理（启用SP模式）
        Args:
            local_cnt_list: 每个worker的local_cnt列表
            sp_cnt_list: 每个worker的sp_cnt列表
            group_sp_cnt_list: 每个worker的group_sp_cnt列表
            cache_seqlens_list: 每个worker的cache_seqlens列表（int类型）
            block_table_list: 每个worker的block_table列表
        Returns:
            所有worker的推理结果列表（与worker顺序一致）
        """
        worker_count = len(self.workers)
        param_lists = [
            local_cnt_list,
            sp_cnt_list,
            group_sp_cnt_list,
            cache_seqlens_list,
            block_table_list,
        ]
        param_names = [
            "local_cnt_list",
            "sp_cnt_list",
            "group_sp_cnt_list",
            "cache_seqlens_list",
            "block_table_list",
        ]

        for param_list, param_name in zip(param_lists, param_names):
            assert (
                len(param_list) == worker_count
            ), f"{param_name}长度({len(param_list)})与worker数量({worker_count})不匹配"

        futures = []
        for i, worker in enumerate(self.workers):
            futures.append(
                worker.run.remote(
                    local_cnt_list[i],
                    sp_cnt_list[i],
                    group_sp_cnt_list[i],
                    cache_seqlens_list[i],
                    block_table_list[i],
                    decode_loopcount,
                )
            )

        return ray.get(futures)

    def step_wo_sp(
        self,
        cache_seqlens_list: List[List[int]],
        block_table_list: List[List[List[int]]],
        decode_loopcount: int,
    ):
        """
        为每个worker分配列表中对应位置的参数并执行推理（禁用SP模式）
        Args:
            cache_seqlens_list: 每个worker的cache_seqlens列表（int类型）
            block_table_list: 每个worker的block_table列表
        Returns:
            所有worker的推理结果列表（与worker顺序一致）
        """
        worker_count = len(self.workers)
        param_lists = [cache_seqlens_list, block_table_list]
        param_names = ["cache_seqlens_list", "block_table_list"]

        for param_list, param_name in zip(param_lists, param_names):
            assert (
                len(param_list) == worker_count
            ), f"{param_name}长度({len(param_list)})与worker数量({worker_count})不匹配"

        futures = []
        for i, worker in enumerate(self.workers):
            futures.append(
                worker.run.remote(
                    cache_seqlens_list[i], block_table_list[i], decode_loopcount
                )
            )

        return ray.get(futures)
