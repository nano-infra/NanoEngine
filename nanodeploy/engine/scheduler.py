from typing import TYPE_CHECKING

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


class UnschedulableRequestError(ValueError):
    """Typed request-local LS ingress rejection."""

    def __init__(self, error, assigned_dp: int, reason: str):
        self.error = error
        self.assigned_dp = assigned_dp
        self.reason = reason
        super().__init__(reason)


# Adapter class for C++ Scheduler to work with Config object
class Scheduler(_CppScheduler):
    def __init__(self, config: Config):
        # C++ Scheduler expects individual parameters, not Config object
        ls_admission_max_tokens_per_pool = config.resolve_ls_admission_max_tokens(
            config.attention_sp
            * config.num_kvcache_blocks
            * config.kvcache_block_size
        )
        config.ls_resolved_admission_max_tokens_per_pool = (
            ls_admission_max_tokens_per_pool
        )
        super().__init__(
            config.engine_id or "",
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
            config.segment_size,
            config.enable_dynamic_sp_size,
            config.use_new_decode_dynamic_sp_scheduler,
            config.dynamic_sp_size_strategy,
            config.dynamic_sp_long_request_threshold,
            config.dynamic_sp_long_request_size,
            config.enable_dynamic_sp_bucket_policy,
            config.dynamic_sp_bucket_policy,
            config.dynamic_sp_attention_cost_a,
            config.dynamic_sp_attention_cost_b,
            config.dynamic_sp_q_cost_a,
            config.dynamic_sp_q_cost_b,
            config.dynamic_sp_res_cost_a,
            config.dynamic_sp_res_cost_b,
            config.dynamic_sp_lse_cost_a,
            config.dynamic_sp_lse_cost_b,
            config.dynamic_sp_q_bytes_per_edge,
            config.dynamic_sp_res_bytes_per_edge,
            config.dynamic_sp_lse_bytes_per_edge,
            config.enable_non_uniform_split,
            config.sp_master_selector,
            config.sp_debug,
            config.fixed_sp_size,
            config.enable_ls_decode_core_scheduler,
            config.ls_decode_initial_kv_dop,
            config.ls_decode_batch_per_master,
            config.ls_decode_enable_memory_scale_up,
            config.scheduler_mode,
            config.ls_kv_consolidation_mode,
            config.ls_kv_consolidation_candidate_util,
            config.ls_kv_consolidation_target_high_watermark,
            config.ls_kv_consolidation_stable_steps,
            config.ls_kv_consolidation_cooldown_steps,
            config.ls_kv_consolidation_check_interval_steps,
            config.ls_kv_consolidation_max_source_blocks_per_event,
            config.ls_decode_enable_future_kv_admission,
            config.ls_max_num_ooe,
            config.ls_running_max_req_size,
            ls_admission_max_tokens_per_pool,
            config.ls_min_comp_bound_decoding_batch_size,
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
        self.ls_admission_max_tokens_per_pool = (
            ls_admission_max_tokens_per_pool
        )

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
