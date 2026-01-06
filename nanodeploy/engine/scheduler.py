from typing import Literal, TYPE_CHECKING

from nanodeploy.config import Config
from nanodeploy.engine.sequence import Sequence
from nanodeploy.logging import get_logger
from nanodeploy._cpp import (
    Scheduler as _CppScheduler,
    SPStateManager as _CppSPStateManager,
    RoutingStrategy as _CppRoutingStrategy,
    DefaultIntDict as _CppDefaultIntDict,
    postprocess_sequences as _cpp_postprocess_sequences
)

if TYPE_CHECKING:
    from nanodeploy.metrics import MetricsManager

logger = get_logger()

SPStateManager = _CppSPStateManager
RoutingStrategy = _CppRoutingStrategy

# Adapter class for C++ Scheduler to work with Config object
class Scheduler(_CppScheduler):
    def __init__(self, config: Config):
        # C++ Scheduler expects individual parameters, not Config object
        super().__init__(
            config.engine_id,
            config.loop_count,
            config.max_num_seqs,
            config.max_num_batched_tokens,
            config.max_num_recv_seqs,
            config.eos,
            config.attention_dp,
            config.attention_sp,
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            config.mode,
            config.reserved_blocks_per_req,
            config.enable_dynamic_sp_size
        )
        # Store config for compatibility
        self.engine_id = config.engine_id
        self.loop_count = config.loop_count
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.attention_dp = config.attention_dp
        self.attention_sp = config.attention_sp
        self.mode = config.mode
        self.routing_strategy = RoutingStrategy[config.routing_strategy]

    def postprocess(
        self,
        dp_sp_seqs: list[list[Sequence]],
        dp_sp_token_ids: list[list[list[int]]],
        metrics_manager: "MetricsManager | None" = None,
        step_duration_ms: float = 0.0,
        loop_count: int = 1,
    ):
        # Use the C++ implementation directly
        return super().postprocess(
            dp_sp_seqs, dp_sp_token_ids, metrics_manager is not None,
            step_duration_ms, loop_count
        )
