import os
import threading
import time as _time
from dataclasses import dataclass
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from dlengine._rust.proto import RunnerIn
from dlengine.config import Config
from dlengine.logging import get_logger, set_log_level
from dlengine.runtime.context.batch import (
    BatchContext,
    get_batch_context,
    set_batch_context,
)
from dlengine.runtime.context.batch_out import get_batch_out_context
from dlengine.runtime.context.cache import (
    CacheContext,
    get_cache_context,
    set_cache_context,
)
from dlengine.runtime.context.cache.hca import get_hca_context
from dlengine.runtime.context.cache.hisparse import (
    allocate_gqa_hot_buffer,
    initialize_hisparse_context,
    initialize_mla_hisparse_cache,
)
from dlengine.runtime.context.cache.plan import CachePlan, gqa_cache_plan
from dlengine.runtime.context.distributed import get_dist_context, set_dist_context
from dlengine.runtime.context.expert import ExpertContext
from dlengine.runtime.context.management import reset_runtime_contexts
from dlengine.runtime.context.parameter import WeightContext, WeightUpdateEngine
from dlengine.runtime.context.peer import PeerAgentContext
from dlengine.runtime.disagg.p2p import get_p2p_cache_transfer
from dlengine.runtime.layers.sampler import Sampler
from dlengine.runtime.models.registry import (
    architecture_loaders,
    architecture_mtp_loaders,
)
from dlengine.runtime.runner.graph_runner import DecodeGraphRunner
from dlengine.runtime.runner.input_preparer import (
    InputPreparer,
    prepare_sample_from_aux,
)
from dlengine.runtime.runner.loader import load_model, load_mtp_model
from dlengine.runtime.runner.mtp_runner import MTPRunner
from dlengine.runtime.runner.runner_config import get_runner_config, set_runner_config
from dlengine.runtime.runner.vision_embed import VisionEmbedManager
from dlengine.utils.network import get_free_port, get_local_ip


def _supports_multi_step_mtp(hardware: str) -> bool:
    return hardware in {"hopper", "blackwell"}


def _wire_mla_hisparse_modules(
    modules, hot_cache: torch.Tensor, layer_id: int, device
) -> int:
    """Wire consecutive paged MLA modules to target/predictor hot slices."""
    for module in modules:
        if not (
            hasattr(module, "k_cache") and getattr(module, "use_paged_kv_cache", True)
        ):
            continue
        if layer_id >= hot_cache.shape[1]:
            raise RuntimeError("MLA HiSparse cache has fewer layers than the model")
        module.k_cache = hot_cache[0, layer_id]
        if hasattr(module, "v_cache"):
            module.v_cache = torch.tensor([], device=device)
        layer_id += 1
    return layer_id


# ─── Per-step host-critical-path timer ─────────────────────────────────────
# Driver enables via ``Config.step_timing=True`` (threaded into RunnerConfig
# in ModelRunner.__init__). Useful for quantifying the gap between
# consecutive cudaGraphLaunch invocations — the GPU-idle window the
# device-signal + pingpong scheduling work would target. Reading
# RunnerConfig (not os.environ) avoids Ray actor env-var mismatch.


class _StepTimer:
    """Lightweight host timer capturing intra-step phase boundaries.

    Phases logged (decode):
        rpc_in       — RunnerIn.aux (deserialize input bytes)
        prep         — prepare_decode_bytes (positions/slot_mapping/block_table)
        forward      — run_model (graph.replay) + GPU-side execution
        sample       — _standard_sample (sampler kernel + GPU execution)
        tail         — token append + IPC wait until next step starts

    Phases include a ``torch.cuda.synchronize()`` at each mark so we
    attribute GPU compute to the phase that issued it. Without this,
    everything looks instant (because graph.replay is async) and the
    real GPU work piles into ``tail`` alongside the IPC wait. Sync
    overhead is ~5-10us per mark → ~30-50us per step, well under 1%
    of a 24ms decode step.

    Total = sum of phases ≈ run_from_bytes wall.
    """

    __slots__ = ("ts",)

    def __init__(self) -> None:
        self.ts: list[tuple[str, int]] = []

    def mark(self, phase: str) -> None:
        # Sync first so the timestamp captures "all GPU work issued up
        # to this point has completed". Cheap when GPU is idle, blocks
        # for the actual compute time when it's busy.
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        self.ts.append((phase, _time.perf_counter_ns()))

    def report(
        self,
        rank: int,
        run_count: int,
        num_seqs: int,
        is_prefill: bool,
        interval: int,
        rank_only: int,
    ) -> None:
        if rank_only >= 0 and rank != rank_only:
            return
        if run_count % interval != 0:
            return
        if len(self.ts) < 2:
            return
        deltas = []
        for i in range(1, len(self.ts)):
            label, t = self.ts[i]
            deltas.append((label, t - self.ts[i - 1][1]))
        total_ns = self.ts[-1][1] - self.ts[0][1]
        total_us = total_ns / 1000
        parts = "  ".join(
            f"{label}={dur/1000:.2f}us({dur/total_ns*100:.0f}%)"
            for label, dur in deltas
        )
        kind = "prefill" if is_prefill else "decode"
        get_logger().info(
            f"[step_timing] r{rank} {kind} step={run_count} "
            f"num_seqs={num_seqs} total={total_us:.2f}us  {parts}"
        )


class _CudaForwardTimer:
    """CUDA-event timer for the GPU-side forward breakdown.

    Records lightweight events on the current stream and reads elapsed_time
    once at report time (a single sync on the final event). Unlike _StepTimer
    it does NOT call torch.cuda.synchronize() at each mark, so it neither
    serializes the pipeline nor distorts the very forward it is measuring —
    elapsed_time reflects the GPU timeline between two events regardless of
    host-side syncs.

    Decode breakdown (deltas between consecutive marks):
        model   — transformer forward / CUDA-graph replay (fwd_start → model)
        logits  — compute_logits / lm_head                (model → logits)
        sample  — sampler kernels                          (logits → sample)

    Note: a CUDA-graph decode replay is a single opaque launch, so the
    transformer cannot be subdivided further from the host; ``model`` is the
    whole captured graph.
    """

    __slots__ = ("events",)

    def __init__(self) -> None:
        self.events: list[tuple[str, "torch.cuda.Event"]] = []

    def mark(self, label: str) -> None:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.events.append((label, ev))

    def report(
        self, rank: int, run_count: int, num_seqs: int, is_prefill: bool
    ) -> None:
        if len(self.events) < 2:
            return
        # elapsed_time needs both events completed; one sync on the last.
        self.events[-1][1].synchronize()
        total_ms = self.events[0][1].elapsed_time(self.events[-1][1])
        parts = []
        for i in range(1, len(self.events)):
            label, ev = self.events[i]
            dt = self.events[i - 1][1].elapsed_time(ev)
            pct = (dt / total_ms * 100) if total_ms > 0 else 0.0
            parts.append(f"{label}={dt:.3f}ms({pct:.0f}%)")
        kind = "prefill" if is_prefill else "decode"
        get_logger().info(
            f"[fwd_timing] r{rank} {kind} step={run_count} "
            f"num_seqs={num_seqs} total={total_ms:.3f}ms  " + "  ".join(parts)
        )


logger = get_logger("DLENGINE")


@dataclass
class _PreparedRun:
    """State produced by prepare_from_bytes and consumed by run_prepared."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    aux: object
    num_seqs: int
    is_prefill: bool
    is_dummy: bool
    has_lazy_verify: bool
    batch_context: BatchContext
    hca_tile_scheduler_metadata: object | None
    runner_config: object
    timer: _StepTimer | None
    fwd_timer: _CudaForwardTimer | None
    gap_start_evt: torch.cuda.Event | None
    gap_report: bool


@ray.remote(num_cpus=0.1, num_gpus=1)
class ModelRunner:
    def __init__(
        self,
        config: Config,
        rank: int,
        defer_dist_init: bool = False,
        debug_env: dict[str, str] | None = None,
    ):
        if debug_env:
            os.environ.update({key: str(value) for key, value in debug_env.items()})

        # Set log level
        if config.log_level:
            set_log_level(config.log_level)

        self.config = config
        os.environ["DLENGINE_USE_FLASHINFER_DECODE"] = (
            "1" if getattr(config, "use_flashinfer_decode", False) else "0"
        )
        os.environ["DLENGINE_USE_FLASHINFER_PREFILL"] = (
            "1" if getattr(config, "use_flashinfer_prefill", False) else "0"
        )
        self.engine_id = self.config.engine_id
        hf_config = config.hf_config
        enable_mla_reference_fallback = getattr(
            config, "enable_mla_reference_fallback", False
        )
        setattr(
            hf_config,
            "enable_mla_reference_fallback",
            enable_mla_reference_fallback,
        )
        for attr in (
            "enable_hisparse",
            "hisparse_device_buffer_size",
            "hisparse_swap_in_block_size",
        ):
            setattr(hf_config, attr, getattr(config, attr, None))
        self.enforce_eager = config.enforce_eager
        if (
            enable_mla_reference_fallback
            and getattr(hf_config, "kv_lora_rank", 0) > 0
            and (
                getattr(hf_config, "kv_lora_rank", 0)
                + getattr(hf_config, "qk_rope_head_dim", 0)
            )
            not in (512, 576)
        ):
            if not self.enforce_eager:
                logger.warning(
                    "Forcing eager decode because MLA reference fallback is "
                    "enabled for unsupported FlashMLA head_dim=%s",
                    getattr(hf_config, "kv_lora_rank", 0)
                    + getattr(hf_config, "qk_rope_head_dim", 0),
                )
            self.enforce_eager = True

        # Disable torch.compile before any compiled layer is built or any
        # lazily-compiled helper runs. Every dlengine call site routes through
        # dlengine.runtime.compile_utils.maybe_compile, which returns the original
        # callable unwrapped when disabled — so torch.compile (and the
        # inductor/triton backend) is never invoked. Must run before model
        # construction in _complete_dist_init().
        if getattr(config, "disable_compile", False):
            from dlengine.runtime.compile_utils import set_compile_disabled

            set_compile_disabled(True)
            logger.info("torch.compile disabled (config.disable_compile=True)")
        self.world_size = config.world_size
        self.rank = rank
        self._dist_initialized = False
        self._dlslime_agent = None
        self._dlslime_alias = None
        self._dlslime_thread = None
        self._dlslime_peer = None
        self.peer_agent_context = None
        self.cache_transfer = get_p2p_cache_transfer()
        self.weight_context = None
        self.weight_update_engine = None
        self.l3_store = None  # Hf3fsL3Store, created in allocate_kvcache when enabled

        logger.debug(f"init ModelRunner, {rank=}, {get_local_ip()=}")

        set_runner_config(
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            dummy_weight=config.dummy_weight,
            dummy_eplb=config.dummy_eplb,
            enable_eplb=config.enable_eplb,
            use_mega_moe=getattr(config, "use_mega_moe", False),
            mega_moe_max_tokens_per_rank=getattr(
                config, "mega_moe_max_tokens_per_rank", 256
            ),
            step_timing=getattr(config, "step_timing", False),
            step_timing_interval=getattr(config, "step_timing_interval", 16),
            step_timing_rank=getattr(config, "step_timing_rank", 0),
            gpu_idle_probe=getattr(config, "gpu_idle_probe", False),
            dlslime_timing=getattr(config, "dlslime_timing", False),
        )

        if defer_dist_init:
            # NanoOps mode: skip heavy init here; caller will invoke
            # init_dist(master_address) after probing the worker node IP.
            logger.info(f"Deferring dist init for rank {rank} (NanoOps mode)")
            return

        self._complete_dist_init()

    # ------------------------------------------------------------------
    # Node probing (called before dist init in NanoOps mode)
    # ------------------------------------------------------------------

    def get_node_info(self):
        """Return (ip, free_port) of the node this worker runs on."""
        return get_local_ip(), get_free_port()

    def init_dist(self, master_address: str):
        """Complete deferred distributed initialization.

        Called by RayExecutor after probing the worker-node IP.
        Must be invoked on ALL ranks simultaneously (collective call).
        """
        self.config.master_address = master_address
        self._complete_dist_init()

    # ------------------------------------------------------------------

    def _complete_dist_init(self):
        """Phase-2 init: process group, contexts, CUDA, model, etc."""
        config = self.config
        hf_config = config.hf_config
        rank = self.rank

        # Per-rank RNG seed. With DP>1, seeding all replicas with the
        # same value makes ``torch.empty_like(...).exponential_(1)`` in
        # the sampler produce byte-identical Gumbel noise on every rank
        # — n>1 sampling collapses to n=1, GRPO advantages become 0,
        # training stalls. Use ``rank`` so each replica has its own
        # stream while the same job is still reproducible from seed 0.
        torch.manual_seed(rank)
        torch.cuda.manual_seed_all(rank)

        if os.getenv("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES") == "1":
            device_count = torch.cuda.device_count()
            if device_count <= 0:
                raise RuntimeError("MegaMoE worker has no visible CUDA devices")
            local_device = rank % device_count
            torch.cuda.set_device(local_device)
            logger.info(
                "MegaMoE CUDA binding: rank=%d local_device=%d visible_devices=%d",
                rank,
                local_device,
                device_count,
            )
        else:
            torch.cuda.set_device(0)

        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            f"tcp://{config.master_address}",
            world_size=self.world_size,
            rank=rank,
        )
        self._dist_initialized = True

        set_dist_context(
            rank=rank,
            world_size=config.world_size,
            attention_dp=config.attention_dp,
            attention_sp=config.attention_sp,
            attention_tp=config.attention_tp,
            ffn_dp=config.ffn_dp,
            ffn_ep=config.ffn_ep,
            ffn_tp=config.ffn_tp,
            pp=config.pp,
        )

        self.default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")

        assert config.attention_sp == 1, "attention_sp > 1 (SP) not supported"
        ep_size = get_dist_context().ffn_ep_world_size

        self.run_count = 0
        self.profiler = None
        self.profiler_trace_dir = None

        # Resolve the hardware decision once at worker startup. Model code still
        # reads the process-local holder during the compatibility migration.
        from dlengine.runtime.layers import set_backend
        from dlengine.runtime.layers.backend_selection import (
            create_backend,
            resolve_backend_selection,
        )
        from dlengine.runtime.models.quant_config import QuantizationConfig as _QC

        _quant_cfg_dict = getattr(hf_config, "quantization_config", None) or {}
        if not isinstance(_quant_cfg_dict, dict):
            _quant_cfg_dict = {}
        try:
            cuda_capability = torch.cuda.get_device_capability()
        except Exception:
            cuda_capability = None
        backend_selection = resolve_backend_selection(
            requested_hardware=getattr(config, "hardware_backend", "auto"),
            requested_attention=getattr(config, "attention_backend", "auto"),
            requested_gdn=getattr(config, "gdn_backend", "auto"),
            cuda_capability=cuda_capability,
            legacy_hardware_backend=os.environ.get("NANO_BACKEND"),
            ref_fallback_allowed=getattr(config, "ref_fallback_allowed", False),
        )
        self.backend = create_backend(
            backend_selection,
            _QC(**_quant_cfg_dict),
        )
        set_backend(self.backend)
        logger.info(
            "Selected runtime backend: hardware=%s attention=%s gdn=%s "
            "source=%s reason=%s ref_fallback=%s (%s)",
            backend_selection.hardware,
            backend_selection.attention,
            backend_selection.gdn,
            backend_selection.hardware_source,
            backend_selection.hardware_reason,
            backend_selection.ref_fallback_allowed,
            backend_selection.fallback_reason,
        )

        model_architecture = hf_config.architectures[0]
        if config.num_speculative_tokens > 1:
            if model_architecture != "GlmMoeDsaForCausalLM":
                raise ValueError(
                    "multi-step MTP is currently supported only for "
                    "GlmMoeDsaForCausalLM"
                )
            if not _supports_multi_step_mtp(backend_selection.hardware):
                raise ValueError(
                    "GLM multi-step MTP currently requires the Hopper or "
                    "Blackwell backend"
                )
        model_loader = architecture_loaders.get(model_architecture)
        if model_loader is None:
            raise ValueError(f"Unsupported architecture {model_architecture}")
        self.model = model_loader()(hf_config)
        self.cache_plan = self._get_model_cache_plan()

        # Warmup ExpertContext for MoE models
        num_total_experts = getattr(hf_config, "num_experts", 0) or getattr(
            hf_config, "n_routed_experts", 0
        )
        if num_total_experts > 0:
            ep_rank = get_dist_context().ffn_ep_rank

            # Check FP8 quantization configs
            quant_config = getattr(hf_config, "quantization_config", None)
            is_fp8 = False
            if quant_config is not None:
                is_fp8 = quant_config.get("quant_method", "") == "fp8"

            num_local_experts = num_total_experts // ep_size
            ExpertContext.get_instance().warmup(
                ep_group=get_dist_context().ffn_ep_group,
                ep_rank=ep_rank,
                ep_size=ep_size,
                num_local_experts=num_local_experts,
                hidden_size=hf_config.hidden_size,
                max_num_sequence=config.max_num_seqs,
                is_fp8=is_fp8,
            )

        if not get_runner_config().dummy_weight:
            load_model(self.model, config.model)

        # --- MTP model initialization ---
        mtp_model = None
        dist_context = get_dist_context()
        owns_mtp = config.num_speculative_tokens > 0 and (
            config.pp == 1 or dist_context.is_last_pp_stage
        )
        if owns_mtp:
            mtp_loader = architecture_mtp_loaders.get(model_architecture)
            if mtp_loader is None:
                raise ValueError(
                    f"MTP not supported for architecture {model_architecture}"
                )
            mtp_cls = mtp_loader()
            mtp_model = mtp_cls(hf_config)
            target_embed = self.model.model.embed_tokens
            if target_embed is not None:
                mtp_model.embed_tokens = target_embed
            if hasattr(mtp_model, "lm_head") and getattr(
                hf_config, "tie_word_embeddings", False
            ):
                mtp_model.lm_head = self.model.lm_head
            if not get_runner_config().dummy_weight:
                load_mtp_model(mtp_model, config.model)
            # Share lm_head weights with MTP shared_head.head —
            # checkpoints either store identical copies or omit the head entirely.
            mtp_layers = (
                mtp_model.layers.values()
                if hasattr(mtp_model.layers, "values")
                else mtp_model.layers
            )
            for _layer in mtp_layers:
                if hasattr(_layer, "shared_head") and hasattr(
                    _layer.shared_head, "head"
                ):
                    _layer.shared_head.head.weight = self.model.lm_head.weight
            logger.info(
                f"MTP model loaded: {mtp_cls.__name__}, "
                f"num_speculative_tokens={config.num_speculative_tokens}"
            )

        dist.barrier()
        self.weight_context = WeightContext()
        self.weight_update_engine = WeightUpdateEngine(self.model, self.weight_context)
        self.sampler = Sampler()
        self.input_preparer = InputPreparer(config)
        self.vision_manager = VisionEmbedManager(hf_config)
        self.mtp_runner = (
            MTPRunner(config, mtp_model, self.sampler) if mtp_model else None
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.preallocate_kvcache()

    def _create_profiler(self, trace_dir: str):
        os.makedirs(trace_dir, exist_ok=True)
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=None,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                dir_name=trace_dir,
                worker_name=f"{self.engine_id}_rank_{self.rank}",
                use_gzip=False,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

    def start_profiler(self, trace_name: str) -> dict:
        """Start a manually delimited CPU/CUDA profiler session."""
        if self.profiler is not None:
            return {
                "rank": self.rank,
                "status": "already_running",
                "trace_name": (
                    Path(self.profiler_trace_dir).name
                    if self.profiler_trace_dir
                    else None
                ),
                "trace_dir": self.profiler_trace_dir,
                "run_count": self.run_count,
            }

        trace_dir = str(
            (Path(self.config.profiler_dir).expanduser().resolve() / trace_name)
        )
        profiler = self._create_profiler(trace_dir)
        profiler.start()
        self.profiler = profiler
        self.profiler_trace_dir = trace_dir
        logger.info(
            "Rank %s: Runtime profiler started at step %s; trace_dir=%s",
            self.rank,
            self.run_count,
            trace_dir,
        )
        return {
            "rank": self.rank,
            "status": "started",
            "trace_name": trace_name,
            "trace_dir": trace_dir,
            "run_count": self.run_count,
        }

    def stop_profiler(self) -> dict:
        """Stop the active profiler and report trace files visible to this worker."""
        if self.profiler is None:
            return {
                "rank": self.rank,
                "status": "not_running",
                "trace_dir": self.profiler_trace_dir,
                "run_count": self.run_count,
                "trace_files": [],
            }

        trace_dir = self.profiler_trace_dir
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        self.profiler.stop()
        self.profiler = None
        trace_files = (
            sorted(str(path) for path in Path(trace_dir).glob("**/*") if path.is_file())
            if trace_dir
            else []
        )
        logger.info(
            "Rank %s: Runtime profiler stopped at step %s; files=%s",
            self.rank,
            self.run_count,
            len(trace_files),
        )
        return {
            "rank": self.rank,
            "status": "stopped",
            "trace_dir": trace_dir,
            "run_count": self.run_count,
            "trace_files": trace_files,
        }

    def _advance_profiler(self) -> None:
        if self.profiler is not None:
            self.profiler.step()

    def num_kvcache_blocks(self):
        return self.config.num_kvcache_blocks

    def num_host_kvcache_blocks(self):
        return get_cache_context().num_host_kvcache_blocks

    def _get_model_cache_plan(self) -> CachePlan:
        cache_plan = getattr(self.config, "cache_plan", None)
        if cache_plan is not None:
            return cache_plan
        get_plan = getattr(self.model, "get_cache_plan", None)
        if callable(get_plan):
            return get_plan()
        logger.warning(
            "Model %s does not declare a cache plan; falling back to GQA",
            type(self.model).__name__,
        )
        return gqa_cache_plan()

    def apply_weight_update(
        self, named_tensors: dict[str, torch.Tensor]
    ) -> dict[str, int]:
        if self.weight_update_engine is None:
            raise RuntimeError("ModelRunner WeightUpdateEngine is not initialized")
        return self.weight_update_engine.apply_named_tensors(named_tensors)

    def pull_and_apply_weights(self, manifest_blob: bytes, train_alias: str) -> dict:
        if self.weight_update_engine is None:
            raise RuntimeError("ModelRunner WeightUpdateEngine is not initialized")
        return self.weight_update_engine.pull_and_apply(manifest_blob, train_alias)

    def allocate_kvcache(self, num_kvcache_blocks: int):
        # Per-rank startup progress. allocate_kvcache runs several collective
        # ops (peer-agent memory registration, CUDA-graph capture, warmup
        # forward with MoE all-to-all); if one rank stalls, every rank hangs.
        # These one-shot INFO lines (tagged with the rank) make it obvious
        # which rank stopped and at which stage. Negligible cost (startup only).
        logger.info(
            f"[startup] r{self.rank} allocate_kvcache begin ({num_kvcache_blocks=})"
        )
        self.config.num_kvcache_blocks = num_kvcache_blocks
        cache_context = get_cache_context()
        pd_mla_hisparse = (
            self.cache_plan.has_hisparse()
            and self.cache_plan.has_mla()
            and bool(self.config.ctrl_address)
        )
        if pd_mla_hisparse:
            cache_context.num_local_kvcache_blocks = num_kvcache_blocks
            cache_context.num_host_kvcache_blocks = num_kvcache_blocks
            cache_context.kv_cache = torch.tensor([], device=cache_context.device)
        else:
            cache_context.allocate_kvcache(num_kvcache_blocks)
        cache_context.allocate_host_kvcache(cache_context.num_host_kvcache_blocks)

        if self.cache_plan.has_hca():
            self._wire_dsv4_caches(cache_context)
        elif not pd_mla_hisparse:
            layer_id = 0
            for module in self.model.modules():
                allocated = False
                if getattr(module, "use_paged_kv_cache", True) is False:
                    if hasattr(module, "k_cache"):
                        module.k_cache = torch.tensor([], device=cache_context.device)
                    if hasattr(module, "v_cache"):
                        module.v_cache = torch.tensor([], device=cache_context.device)
                    continue
                if hasattr(module, "k_cache"):
                    module.k_cache = cache_context.kv_cache[0][layer_id]
                    allocated = True
                if hasattr(module, "v_cache"):
                    if cache_context.kv_cache.size(0) > 1:
                        module.v_cache = cache_context.kv_cache[1][layer_id]
                    else:
                        module.v_cache = torch.tensor([], device=cache_context.device)
                    allocated = True
                if allocated:
                    layer_id += 1

            # The single GLM NextN layer is a real attention layer. Recurrent
            # MTP must retain its own MLA KV instead of running every draft as
            # an isolated one-token prefill. Its cache slices live directly
            # after the target-model slices so scheduler block IDs and RDMA
            # migration remain shared.
            if self.mtp_runner is not None:
                for module in self.mtp_runner.mtp_model.modules():
                    allocated = False
                    if hasattr(module, "k_cache"):
                        module.k_cache = cache_context.kv_cache[0][layer_id]
                        allocated = True
                    if hasattr(module, "v_cache"):
                        if cache_context.kv_cache.size(0) > 1:
                            module.v_cache = cache_context.kv_cache[1][layer_id]
                        else:
                            module.v_cache = torch.tensor(
                                [], device=cache_context.device
                            )
                        allocated = True
                    if allocated:
                        layer_id += 1
        else:
            for module in self.model.modules():
                if hasattr(module, "k_cache"):
                    module.k_cache = torch.tensor([], device=cache_context.device)
                if hasattr(module, "v_cache"):
                    module.v_cache = torch.tensor([], device=cache_context.device)

        # Allocate NSA indexer cache (V3.2 only)
        if self.cache_plan.has_indexer() and cache_context.index_head_dim > 0:
            cache_context.allocate_indexer_cache(self.config.hf_config)
            # Wire indexer cache to each layer's Indexer module
            for module in self.model.modules():
                if hasattr(module, "indexer") and module.indexer is not None:
                    module.indexer.indexer_cache = cache_context.indexer_cache
            if self.mtp_runner is not None:
                for module in self.mtp_runner.mtp_model.modules():
                    if hasattr(module, "indexer") and module.indexer is not None:
                        module.indexer.indexer_cache = cache_context.indexer_cache

        if self.cache_plan.has_hisparse():
            hisparse_ctx = initialize_hisparse_context(
                self.config.max_num_seqs,
                cache_context.device,
                self.config.hisparse_device_buffer_size,
            )
            if self.cache_plan.has_gqa():
                total_tokens = hisparse_ctx.tokens_per_seq * max(
                    1, hisparse_ctx.max_num_seqs
                )
                for module in self.model.modules():
                    if not (
                        hasattr(module, "hisparse_k_cache")
                        and hasattr(module, "hisparse_v_cache")
                    ):
                        continue
                    num_kv_heads = getattr(
                        module, "num_kv_heads", cache_context.num_local_kv_heads
                    )
                    head_dim = getattr(module, "head_dim", cache_context.head_dim)
                    module.hisparse_k_cache = torch.empty(
                        total_tokens,
                        num_kv_heads,
                        head_dim,
                        dtype=cache_context.dtype,
                        device=cache_context.device,
                    )
                    module.hisparse_v_cache = torch.empty_like(module.hisparse_k_cache)
            else:
                if self.config.ctrl_address:
                    if cache_context.host_kv_cache is None:
                        raise RuntimeError(
                            "PD MLA HiSparse requires decode host KV cache; set "
                            "host_utilization_per_device high enough for cold KV"
                        )
                    # Release the provisional full GPU cache before allocating
                    # the bounded hot tier; retaining both can defeat HiSparse
                    # precisely when GPU memory is tight.
                    for module in self.model.modules():
                        if hasattr(module, "k_cache") and getattr(
                            module, "use_paged_kv_cache", True
                        ):
                            module.k_cache = torch.tensor(
                                [], device=cache_context.device
                            )
                            if hasattr(module, "v_cache"):
                                module.v_cache = torch.tensor(
                                    [], device=cache_context.device
                                )
                    if self.mtp_runner is not None:
                        for module in self.mtp_runner.mtp_model.modules():
                            if hasattr(module, "k_cache") and getattr(
                                module, "use_paged_kv_cache", True
                            ):
                                module.k_cache = torch.tensor(
                                    [], device=cache_context.device
                                )
                                if hasattr(module, "v_cache"):
                                    module.v_cache = torch.tensor(
                                        [], device=cache_context.device
                                    )
                    cache_context.kv_cache = torch.tensor(
                        [], device=cache_context.device
                    )
                    torch.cuda.empty_cache()
                    hot_cache = initialize_mla_hisparse_cache(
                        cache_context.host_kv_cache,
                        max_num_seqs=self.config.max_num_seqs,
                        device_buffer_size=self.config.hisparse_device_buffer_size,
                    )
                    layer_id = 0
                    layer_id = _wire_mla_hisparse_modules(
                        self.model.modules(),
                        hot_cache,
                        layer_id,
                        cache_context.device,
                    )
                    if self.mtp_runner is not None:
                        layer_id = _wire_mla_hisparse_modules(
                            self.mtp_runner.mtp_model.modules(),
                            hot_cache,
                            layer_id,
                            cache_context.device,
                        )
                    if layer_id != hot_cache.shape[1]:
                        raise RuntimeError(
                            "MLA HiSparse cache/model layer mismatch: "
                            f"wired={layer_id}, allocated={hot_cache.shape[1]}"
                        )
                    # Drop the full logical GPU MLA allocation. Scheduler block
                    # IDs continue to name cold host pages; model layers and
                    # CacheContext now expose only the bounded hot tier.
                    cache_context.kv_cache = hot_cache
                elif not self.config.dummy_prefill:
                    raise RuntimeError(
                        "HiSparse MLA requires PD host migration or dummy_prefill=True"
                    )
                with torch.no_grad():
                    cache_context.kv_cache.zero_()
                    if cache_context.indexer_cache is not None:
                        cache_context.indexer_cache.buffer.zero_()
            logger.info(
                "HiSparse initialized: device_buffer_size_per_seq=%s, "
                "total_device_tokens=%s, swap_in_block_size=%s",
                self.config.hisparse_device_buffer_size,
                hisparse_ctx.tokens_per_seq * max(1, hisparse_ctx.max_num_seqs),
                self.config.hisparse_swap_in_block_size,
            )

        wire_shared_kv_caches = getattr(self.model, "wire_shared_kv_caches", None)
        if callable(wire_shared_kv_caches):
            wire_shared_kv_caches()

        # Register memory regions after KV/indexer tensors exist. The PeerAgent
        # itself is started during preallocate_kvcache().
        logger.info(f"[startup] r{self.rank} register_peer_agent_memory_regions begin")
        self.cache_transfer.register_peer_agent_memory_regions(mode=self.config.mode)
        logger.info(f"[startup] r{self.rank} register_peer_agent_memory_regions done")

        # L3 (3FS) tiered KV cache: build the per-worker USRBIO store now that
        # the kv_cache tensor exists. Inert unless config.l3_enable.
        if getattr(self.config, "l3_enable", False):
            try:
                from dlengine.runtime.disagg.storage.l3_hf3fs import Hf3fsL3Store

                self.l3_store = Hf3fsL3Store(
                    cache_context,
                    mountpoint=self.config.l3_mountpoint,
                    l3_dir=self.config.l3_dir,
                    staging_blocks=self.config.l3_staging_blocks,
                    rank=self.rank,
                )
            except Exception as e:
                logger.error(f"[L3] failed to init Hf3fsL3Store, disabling: {e}")
                self.l3_store = None

        if not self.enforce_eager:
            logger.info(f"[startup] r{self.rank} cudagraph capture begin")
            self._init_graph_runners()
            logger.info(f"[startup] r{self.rank} cudagraph capture done")
        torch.set_default_device("cpu")
        torch.set_default_dtype(self.default_dtype)
        logger.info(f"[startup] r{self.rank} warmup_model begin")
        self.warmup_model()
        logger.info(f"[startup] r{self.rank} warmup_model done")

    def _wire_dsv4_caches(self, cache_context):
        """Wire DSv4 FP8 paged SWA cache + compressed caches to attention layers."""
        from dlengine.runtime.models.deepseek_v4.deepseek_v4 import DeepseekV4Attention

        # Collect compress_ratios from model layers
        compress_ratios = []
        layer_id = 0
        for module in self.model.modules():
            if isinstance(module, DeepseekV4Attention):
                compress_ratios.append(getattr(module, "compress_ratio", 0))
                # Wire SWA paged cache (per layer slice)
                module.swa_cache = cache_context.kv_cache[layer_id]
                # Keep old k_cache/v_cache as empty for backward compat
                module.k_cache = torch.tensor([], device=cache_context.device)
                module.v_cache = torch.tensor([], device=cache_context.device)
                layer_id += 1

        # Pool sizes from config (0 = derive worst case)
        pool_pages_per_ratio = {}
        if self.config.dsv4_compressed_pool_pages_ratio4 > 0:
            pool_pages_per_ratio[4] = self.config.dsv4_compressed_pool_pages_ratio4
        if self.config.dsv4_compressed_pool_pages_ratio128 > 0:
            pool_pages_per_ratio[128] = self.config.dsv4_compressed_pool_pages_ratio128

        # Allocate compressed caches for layers with compress_ratio > 0
        cache_context.allocate_dsv4_compressed_caches(
            compress_ratios,
            max_num_seqs=self.config.max_num_seqs,
            max_model_len=self.config.max_model_len,
            pool_pages_per_ratio=pool_pages_per_ratio,
        )

        # S2.2: allocate flat compressor scratch state (per ratio, all layers).
        # Layers will use views into these buffers for RDMA-friendly migration.
        cache_context.allocate_dsv4_compressor_state(
            compress_ratios=compress_ratios,
            head_dim=512,  # DSv4 fixed head dim
            max_num_seqs=self.config.max_num_seqs,
        )

        # Wire compressed caches to layers and initialize tensorized compressor state
        layer_id = 0
        for module in self.model.modules():
            if isinstance(module, DeepseekV4Attention):
                if layer_id in cache_context.dsv4_compressed_caches:
                    module.compressed_cache = cache_context.dsv4_compressed_caches[
                        layer_id
                    ]
                else:
                    module.compressed_cache = None
                # Initialize tensorized compressor state — pass views into the
                # per-ratio flat tensors when available.
                if hasattr(module, "compressor") and module.compress_ratio > 0:
                    ratio = module.compress_ratio
                    ratio_layer_idx = cache_context.dsv4_layer_to_ratio_idx.get(
                        layer_id
                    )
                    kv_view = score_view = counts_view = None
                    if (
                        ratio_layer_idx is not None
                        and ratio in cache_context.dsv4_compressor_kv_flat
                    ):
                        kv_view = cache_context.dsv4_compressor_kv_flat[ratio][
                            ratio_layer_idx
                        ]
                        score_view = cache_context.dsv4_compressor_score_flat[ratio][
                            ratio_layer_idx
                        ]
                        counts_view = cache_context.dsv4_compressor_counts_flat[ratio][
                            ratio_layer_idx
                        ]
                    module.compressor.init_tensorized_state(
                        max_slots=self.config.max_num_seqs,
                        device=cache_context.device,
                        kv_view=kv_view,
                        score_view=score_view,
                        counts_view=counts_view,
                    )
                layer_id += 1

    def get_peer_agent_addr(self) -> str | None:
        """Return the peer agent address for this rank."""
        return self.cache_transfer.get_peer_agent_addr()

    def get_peer_agent_placement(self) -> dict | None:
        """Return normalized Fabric placement for this worker, when available."""
        if self.peer_agent_context is None:
            return None
        return self.peer_agent_context.local_placement()

    def start_dlslime_server(self, driver_alias: str) -> str:
        """Start a dedicated DLSLime server for executor transport."""
        if self._dlslime_alias is not None:
            if self._dlslime_peer != driver_alias:
                raise RuntimeError(
                    "DLSLime server already initialized for a different driver "
                    f"({self._dlslime_peer} != {driver_alias})"
                )
            return self._dlslime_alias

        if not self.config.ctrl_address:
            raise RuntimeError(
                "executor_backend='dlslime' requires ctrl_address to be set"
            )

        try:
            import dlslime
            from dlslime.rpc import serve
        except ImportError as exc:
            raise RuntimeError(
                "DLSLime transport requires the optional 'dlslime' dependency"
            ) from exc

        from dlengine.executor.dlslime_protocol import ModelRunnerRpcService

        available_nics = dlslime.available_nic()
        if not available_nics:
            raise RuntimeError("No available NICs found for DLSLime worker agent")

        agent_alias = f"{self.engine_id}:rpc:{self.rank}"
        device = available_nics[get_dist_context().local_rank % len(available_nics)]
        self._dlslime_agent = dlslime.start_peer_agent(
            ctrl_url=self.config.ctrl_address,
            alias=agent_alias,
            device=device,
            scope=self.config.ctrl_scope,
        )
        self._dlslime_qp_num = int(os.environ.get("SLIME_QP_NUM", 1))
        service = ModelRunnerRpcService(self)

        def _serve_loop():
            conn = self._dlslime_agent.connect_to(
                driver_alias,
                ib_port=1,
                qp_num=self._dlslime_qp_num,
            )
            conn.wait(timeout=30)
            serve(self._dlslime_agent, service, driver_alias)

        self._dlslime_thread = threading.Thread(
            target=_serve_loop,
            daemon=True,
            name=f"dlslime-rank-{self.rank}",
        )
        self._dlslime_thread.start()
        self._dlslime_alias = agent_alias
        self._dlslime_peer = driver_alias
        logger.info(f"Rank {self.rank}: DLSLime server ready at {self._dlslime_alias}")
        return self._dlslime_alias

    def p2p_disconnect(self, remote_engine_id: str):
        return self.cache_transfer.p2p_disconnect(remote_engine_id)

    def get_num_connected_peers(self):
        peer_context = self.cache_transfer.peer_agent_context
        if peer_context is None:
            return 0
        return len(peer_context.connected_peers)

    def exit(self):
        if not self.enforce_eager:
            del self.decode_graph_runner
            if self.mtp_runner is not None:
                self.mtp_runner.cleanup()
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            min(self.config.max_num_batched_tokens, 16384),
            min(self.config.max_model_len, 8192),
        )
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        # Pre-compile the prefill MoE DeepGEMM kernels before serving so the
        # first real prefill doesn't JIT-compile inside the forward (which
        # stalls this rank while peers time out in the DeepEP combine
        # collective). Best-effort: never fatal.
        # GLM-5.3 uses a padded expert layout that the installed Blackwell
        # DeepGEMM build rejects during its synthetic warmup.  Its runtime
        # expert path already has a correctness fallback, so avoid launching
        # the incompatible probe (a CUDA launch failure poisons the context).
        arch = (getattr(self.config.hf_config, "architectures", None) or [""])[0]
        if arch != "Glm5NextForConditionalGeneration":
            try:
                self._warmup_deep_gemm_moe(max_num_batched_tokens)
            except Exception as e:  # pragma: no cover - warmup must never crash boot
                logger.warning(f"[startup] r{self.rank} deep_gemm MoE warmup skipped: {e}")
        else:
            logger.info("[startup] GLM-5.3 DeepGEMM MoE warmup disabled; using runtime fallback")
        # Empty warmup batch, built without exposing Sequence serializers.
        warmup_data = RunnerIn.dummy("", 0, True).to_bytes()
        self.run_from_bytes(warmup_data, True)
        torch.cuda.empty_cache()

    def _warmup_deep_gemm_moe(self, max_num_batched_tokens: int):
        """Pre-compile DeepGEMM grouped-GEMM kernels for the prefill MoE path.

        The prefill expert path (``fused_moe_v3`` / ``fused_moe_v3_bf16``) uses
        DeepGEMM's *contiguous* grouped GEMM, whose JIT-compiled kernel + block
        config is selected by the token count. The existing dummy warmup only
        exercises a single 1-token prefill, so the large-M kernels real prefills
        need are JIT-compiled lazily inside the first serving forward. That
        stalls the compiling rank for seconds while the other ranks sit in the
        DeepEP ``combine`` collective until their watchdog fires, surfacing as
        ``CUDA error: unspecified launch failure`` (DeepEP intranode.cu).

        We warm the kernels here by replaying the post-dispatch compute on
        synthetic inputs across a sweep of token-count buckets. ``fused_moe_v3``
        is pure local DeepGEMM compute (no DeepEP collective), so this is
        collective-safe and runs independently on every rank.
        """
        import os

        if os.environ.get("DLENGINE_DISABLE_DEEPGEMM_WARMUP", "") == "1":
            return
        try:
            import deep_gemm  # noqa: F401
        except ImportError:
            return

        # Locate a routed-experts module (all MoE layers share weight shapes,
        # so warming one compiles the kernels used by every layer).
        experts = None
        for module in self.model.modules():
            if (
                hasattr(module, "gate_up_proj")
                and hasattr(module, "down_proj")
                and hasattr(module, "top_k")
                and hasattr(module, "num_local_experts")
            ):
                experts = module
                break
        # Only the EP (DeepGEMM grouped) path needs this; EP==1 uses a different
        # local path that doesn't JIT large grouped GEMMs.
        if experts is None or int(getattr(experts, "ep_size", 1)) <= 1:
            return
        # MegaMoE uses its own packed-MXFP4 DeepGEMM kernels and is already
        # compiled by graph/model warmup. The legacy BF16/FP8 grouped-GEMM
        # warmup below is incompatible with its packed weights.
        from dlengine.runtime.layers.backends.experts.mega_moe import MegaMoEExperts

        if isinstance(experts, MegaMoEExperts):
            return

        from dlengine.runtime.kernel.triton.hopper.fused_moe_v3 import (
            fused_moe_v3,
            fused_moe_v3_bf16,
        )

        E = int(experts.num_local_experts)
        top_k = int(experts.top_k)
        K = int(experts.gate_up_proj.size(2))  # hidden_size
        if E <= 0 or top_k <= 0:
            return

        # ep_scatter requires all_tokens % 128 == 0; keep per-expert counts a
        # multiple of 128 (also the contiguous-layout alignment) and the total
        # token budget under the per-step cap.
        cap = max(int(max_num_batched_tokens), E * 128)
        cs = [c for c in (128, 256, 512, 1024, 2048) if E * c <= cap]
        if not cs:
            cs = [128]

        is_fp8 = bool(getattr(experts, "is_fp8", False))
        swiglu_limit = float(getattr(experts, "_swiglu_limit_runtime", float("inf")))
        dev = experts.gate_up_proj.device

        logger.info(
            f"[startup] r{self.rank} deep_gemm MoE warmup begin "
            f"(fp8={is_fp8} E={E} top_k={top_k} buckets={[E * c for c in cs]})"
        )
        with torch.inference_mode():
            for c in cs:
                all_tokens = E * c
                if all_tokens % top_k != 0:
                    # need an integer #tokens to reshape into [m, top_k]
                    continue
                m_orig = all_tokens // top_k
                num_recv = [c] * E
                # The token->expert histogram must equal num_recv exactly, or
                # ep_scatter writes outside each expert's region. Assign each
                # expert exactly c rows.
                topk_idx = (
                    torch.arange(E, device=dev)
                    .repeat_interleave(c)
                    .reshape(m_orig, top_k)
                    .to(torch.int64)
                )
                topk_weights = torch.ones(
                    (m_orig, top_k), dtype=torch.float32, device=dev
                )
                try:
                    if is_fp8:
                        x_fp8 = torch.zeros(
                            (m_orig, K), dtype=torch.float8_e4m3fn, device=dev
                        )
                        x_scale = torch.ones(
                            (m_orig, K // 128), dtype=torch.float32, device=dev
                        )
                        fused_moe_v3(
                            (x_fp8, x_scale),
                            topk_idx,
                            topk_weights,
                            (experts.gate_up_proj, experts.gate_up_scale_inv),
                            (experts.down_proj, experts.down_scale_inv),
                            num_recv,
                            swiglu_limit=swiglu_limit,
                        )
                    else:
                        x = torch.zeros((m_orig, K), dtype=torch.bfloat16, device=dev)
                        fused_moe_v3_bf16(
                            x,
                            topk_idx,
                            topk_weights,
                            experts.gate_up_proj,
                            experts.down_proj,
                            num_recv,
                            swiglu_limit=swiglu_limit,
                        )
                except Exception as e:
                    logger.warning(
                        f"[startup] r{self.rank} deep_gemm MoE warmup bucket "
                        f"all_tokens={all_tokens} failed: {e}"
                    )
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info(f"[startup] r{self.rank} deep_gemm MoE warmup done")

    def preallocate_kvcache(self):
        config = self.config
        hf_config = config.hf_config

        cache_plan = self.cache_plan
        mode = cache_plan.cache_mode()
        kv_lora_rank = (
            hf_config.kv_lora_rank if hasattr(hf_config, "kv_lora_rank") else 0
        )
        qk_rope_head_dim = (
            hf_config.qk_rope_head_dim if hasattr(hf_config, "qk_rope_head_dim") else 0
        )

        # For linear-attention hybrids (Qwen3.5-MoE), only full_attention
        # layers need paged KV cache. Gemma4 sliding/full layers are both GQA
        # attention layers, so they each need a cache slice.
        layer_types = getattr(hf_config, "layer_types", None)
        arch = (getattr(hf_config, "architectures", None) or [""])[0]
        from dlengine.runtime.models.pp_utils import (
            get_gemma4_pp_layer_range,
            get_pp_layer_range,
        )

        if arch in ("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"):
            pp_start, pp_end = get_gemma4_pp_layer_range(hf_config)
        else:
            pp_start, pp_end = get_pp_layer_range(hf_config.num_hidden_layers)
        local_layer_types = (
            layer_types[pp_start:pp_end] if layer_types is not None else None
        )
        if (
            arch in ("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration")
            and layer_types is not None
        ):
            first_shared = hf_config.num_hidden_layers - getattr(
                hf_config, "num_kv_shared_layers", 0
            )
            first_shared = max(0, first_shared)
            if cache_plan.has_hisparse() and cache_plan.has_gqa():
                num_kv_layers = sum(
                    1
                    for i, lt in enumerate(layer_types[pp_start:pp_end], start=pp_start)
                    if i < first_shared and lt == "full_attention"
                )
            else:
                num_kv_layers = sum(
                    1
                    for i, _lt in enumerate(
                        layer_types[pp_start:pp_end], start=pp_start
                    )
                    if i < first_shared
                )
        elif local_layer_types is not None and any(
            lt == "linear_attention" for lt in local_layer_types
        ):
            # Qwen uses full_attention; GLM-5.3 calls its paged MLA layers
            # deepseek_sparse_attention.  Both are the non-linear cache side.
            num_kv_layers = sum(1 for lt in local_layer_types if lt != "linear_attention")
        else:
            # With pipeline parallelism each stage owns only a contiguous slice
            # of the decoder layers, so it allocates KV cache for just those
            # local layers. get_pp_layer_range returns (0, num_hidden_layers)
            # when pp == 1, preserving the original behaviour.
            num_kv_layers = pp_end - pp_start

        # One physical predictor layer is reused for all five GLM draft steps.
        # It nevertheless needs one persistent MLA/DSA cache slice. Count the
        # physical predictor layers, never num_speculative_tokens.
        num_mtp_kv_layers = 0
        if self.mtp_runner is not None and cache_plan.has_mla():
            num_mtp_kv_layers = max(
                1, int(getattr(self.mtp_runner.mtp_model, "num_mtp_layers", 1))
            )
        self._num_target_kv_layers = num_kv_layers
        self._num_mtp_kv_layers = num_mtp_kv_layers
        num_kv_layers += num_mtp_kv_layers

        # If ctrl_address is provided, fetch engine_id from NanoCtrl
        engine_id = config.engine_id
        if config.ctrl_address and not engine_id:
            engine_id = _get_engine_id_from_ctrl(
                config.ctrl_address, config.host, config.port
            )

        # Enable FP8 KV cache for sparse attention (V3.2).
        # The reference fallback is correctness-only and cannot read the FP8
        # packed cache, so disabling FP8 cache for unsupported shapes is opt-in.
        index_head_dim = getattr(hf_config, "index_head_dim", 0)
        mla_head_dim = kv_lora_rank + qk_rope_head_dim
        flash_mla_supported = (
            mla_head_dim in (512, 576)
            and os.environ.get("DLENGINE_FORCE_MLA_REFERENCE", "0") != "1"
        )
        enable_mla_reference_fallback = getattr(
            config, "enable_mla_reference_fallback", False
        )
        is_fp8_kvcache = (
            cache_plan.has_indexer() and index_head_dim > 0
        ) and not getattr(config, "disable_nsa", False)
        # The native MLA backend consumes the logical BF16 KV layout.
        # Packed FP8 caches include scale metadata and are not ABI-compatible.
        from dlengine.runtime.layers import get_backend

        hardware_backend = getattr(get_backend(), "hardware_backend", "")
        if is_fp8_kvcache and enable_mla_reference_fallback and not flash_mla_supported:
            logger.warning(
                "Disabling FP8 MLA KV cache because MLA reference fallback is "
                "enabled for unsupported FlashMLA head_dim=%s.",
                mla_head_dim,
            )
            is_fp8_kvcache = False
        raw_fp8_mla_layout = is_fp8_kvcache and hardware_backend == "blackwell"
        if cache_plan.has_hisparse() and cache_plan.has_mla() and not is_fp8_kvcache:
            raise RuntimeError("HiSparse requires FP8 MLA KV cache")

        head_dim = getattr(hf_config, "head_dim", None) or (
            hf_config.hidden_size // hf_config.num_attention_heads
        )
        if (
            arch in ("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration")
            and cache_plan.has_hisparse()
            and cache_plan.has_gqa()
            and getattr(hf_config, "global_head_dim", None)
        ):
            # Gemma4 uses 256-dim sliding-window heads and 512-dim global heads.
            # In HiSparse mode paged KV is only used by non-shared full layers;
            # SWA layers use the hot buffer, so the paged cache shape must match
            # the full-attention global head dim.
            head_dim = hf_config.global_head_dim
        # Reserve memory for GDN linear-attention state buffers (allocated below
        # via allocate_gdn_states) so KV-cache sizing stays within the
        # utilization target on hybrid models.
        reserved_state_bytes = 0
        gdn_cache_slots = max(0, getattr(config, "gdn_state_cache_slots", 0))
        if cache_plan.has_gdn() and local_layer_types is not None:
            reserved_state_bytes = CacheContext.estimate_gdn_state_bytes(
                hf_config,
                local_layer_types,
                config.max_num_seqs,
                need_backup=config.num_speculative_tokens > 0,
                cache_slots=gdn_cache_slots,
                attention_tp=config.attention_tp,
            )
        if getattr(config, "use_flashinfer_decode", False):
            reserved_state_bytes += 128 * 1024 * 1024
        if getattr(config, "use_flashinfer_prefill", False):
            reserved_state_bytes += 128 * 1024 * 1024
        cache_context = set_cache_context(
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=head_dim,
            block_size=config.kvcache_block_size,
            num_hidden_layers=num_kv_layers,
            attention_tp=config.attention_tp,
            gpu_memory_utilization=config.gpu_memory_utilization,
            gpu_memory_limit_gb=config.gpu_memory_limit_gb,
            host_utilization_per_device=config.host_utilization_per_device,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            index_head_dim=index_head_dim,
            is_fp8_kvcache=is_fp8_kvcache,
            raw_fp8_mla_layout=raw_fp8_mla_layout,
            device=torch.get_default_device(),
            dtype=torch.get_default_dtype(),
            mode=mode,
            ctrl_address=config.ctrl_address,
            ctrl_scope=config.ctrl_scope,
            engine_id=engine_id,
            architecture=(getattr(hf_config, "architectures", None) or [""])[0],
            enable_hisparse=bool(config.enable_hisparse),
            max_num_seqs=config.max_num_seqs,
            hisparse_device_buffer_size=config.hisparse_device_buffer_size,
            reserved_state_bytes=reserved_state_bytes,
        )
        if (
            self.mtp_runner is not None
            and config.num_speculative_tokens > 1
            and config.mode in ("prefill", "decode")
        ):
            cache_context.allocate_mtp_handoff(config.num_speculative_tokens)
        config.num_kvcache_blocks = cache_context.num_local_kvcache_blocks
        self._sync_cache_plan_from_context(
            cache_plan,
            cache_context,
            num_kv_layers,
            head_dim,
            index_head_dim,
            reserved_state_bytes,
            gdn_cache_slots,
        )

        # Allocate GDN state buffers for linear_attention layers
        if cache_plan.has_gdn() and local_layer_types is not None:
            cache_context.allocate_gdn_states(
                hf_config,
                local_layer_types,
                config.max_num_seqs,
                need_backup=config.num_speculative_tokens > 0,
                cache_slots=gdn_cache_slots,
            )

        self._init_peer_agent_context(cache_context)

    def _sync_cache_plan_from_context(
        self,
        cache_plan: CachePlan,
        cache_context: CacheContext,
        num_kv_layers: int,
        head_dim: int,
        index_head_dim: int,
        reserved_state_bytes: int,
        gdn_cache_slots: int,
    ) -> None:
        max_blocks_per_seq = (
            self.config.max_model_len + cache_context.block_size - 1
        ) // cache_context.block_size

        if cache_plan.has_gqa():
            gqa = cache_plan.gqa
            gqa.num_pages = cache_context.num_local_kvcache_blocks
            gqa.page_size = cache_context.block_size
            gqa.max_blocks_per_seq = max_blocks_per_seq
            gqa.num_layers = num_kv_layers
            gqa.num_kv_heads = cache_context.num_kv_heads
            gqa.head_dim = head_dim
            cache_plan.gqa = gqa

        if cache_plan.has_mla():
            mla = cache_plan.mla
            mla.num_pages = cache_context.num_local_kvcache_blocks
            mla.page_size = cache_context.block_size
            mla.max_blocks_per_seq = max_blocks_per_seq
            mla.num_layers = num_kv_layers
            mla.kv_lora_rank = cache_context.kv_lora_rank
            mla.qk_rope_head_dim = cache_context.qk_rope_head_dim
            mla.head_dim = head_dim
            cache_plan.mla = mla

        if cache_plan.has_indexer():
            indexer = cache_plan.indexer
            indexer.num_pages = cache_context.num_local_kvcache_blocks
            indexer.page_size = cache_context.block_size
            indexer.max_blocks_per_seq = max_blocks_per_seq
            indexer.index_head_dim = index_head_dim
            if index_head_dim > 0:
                indexer.bytes_per_token = index_head_dim + index_head_dim // 128 * 4
            cache_plan.indexer = indexer

        if cache_plan.has_gdn():
            gdn = cache_plan.gdn
            gdn.state_slots = self.config.max_num_seqs + gdn_cache_slots
            gdn.state_bytes = reserved_state_bytes
            cache_plan.gdn = gdn

        if cache_plan.has_hisparse():
            hisparse = cache_plan.hisparse
            hisparse.max_num_seqs = self.config.max_num_seqs
            hisparse.device_buffer_size = self.config.hisparse_device_buffer_size
            if cache_plan.has_gqa():
                hisparse.swap_in_block_size = int(
                    getattr(self.config.hf_config, "sliding_window", 0)
                    or self.config.hisparse_swap_in_block_size
                )
            else:
                hisparse.swap_in_block_size = self.config.hisparse_swap_in_block_size
            hisparse.dummy_slot = self.config.max_num_seqs
            cache_plan.hisparse = hisparse

    def _init_peer_agent_context(self, cache_context):
        """Start the worker-owned PeerAgent and inject it into RDMA users."""
        self.peer_agent_context = PeerAgentContext.start_for_cache_context(
            cache_context,
            rank=dist.get_rank(),
        )
        self.cache_transfer.set_peer_agent_context(self.peer_agent_context)
        cache_context.peer_context = self.peer_agent_context
        cache_context.peer_fabric_enabled = bool(
            self.peer_agent_context is not None
            and self.peer_agent_context.supports_cuda_fabric()
        )
        if cache_context.peer_fabric_enabled:
            logger.info("PeerAgent CUDA Fabric cache allocation enabled")
        if self.weight_context is not None:
            self.weight_context.set_peer_agent_context(self.peer_agent_context)
        if self.peer_agent_context is not None:
            self.vision_manager.set_peer_agent_context(self.peer_agent_context)

    @torch.inference_mode()
    def _mark_fwd(self, label: str) -> None:
        """Record a CUDA-event mark on the active forward timer, if any."""
        ft = getattr(self, "_fwd_timer", None)
        if ft is not None:
            ft.mark(label)

    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        # Pipeline parallelism: non-final stages run their local decoder layers
        # and send the residual stream to the next stage inside model.forward.
        # They produce no logits, so short-circuit before compute_logits.
        is_last_pp_stage = get_dist_context().is_last_pp_stage
        flashinfer_decode_eager = (
            not is_prefill
            and os.environ.get("DLENGINE_FLASHINFER_EAGER_DECODE", "0") == "1"
        )
        if (
            is_prefill
            or self.enforce_eager
            or flashinfer_decode_eager
            or input_ids.size(0) > 512
        ):
            context = get_batch_context()
            inputs_embeds = None
            if is_prefill and self.vision_manager.has_embeds:
                logger.info(
                    f"[RUN_MODEL] Injecting vision embeds for prefill, "
                    f"input_ids.shape={input_ids.shape}"
                )
                inputs_embeds = self.vision_manager.inject(
                    input_ids, self.model.model.embed_tokens
                )
                self.vision_manager.clear()
            elif is_prefill and not context.is_dummy:
                logger.debug(
                    f"[RUN_MODEL] Prefill WITHOUT vision embeds, "
                    f"input_ids.shape={input_ids.shape}"
                )
            if inputs_embeds is not None:
                hidden = self.model(input_ids, positions, inputs_embeds=inputs_embeds)
            else:
                hidden = self.model(input_ids, positions)
            if is_prefill:
                ExpertContext.get_instance().transition_to_low_latency()
            self._mark_fwd("model")
            # Non-final pipeline stages already forwarded the residual stream to
            # the next stage; there is nothing to sample here.
            if not is_last_pp_stage:
                return None
            if self.mtp_runner is not None:
                self.mtp_runner.last_hidden = hidden
            logits = self.model.compute_logits(hidden)
            self._mark_fwd("logits")
            return logits
        else:
            context = get_batch_context()

            # Linear lazy-verify path (seqlen_q=N+1): dedicated graph runner
            if (
                context.num_tokens_per_seq > 1
                and self.mtp_runner is not None
                and self.mtp_runner.lv_graph_runner is not None
            ):
                outputs = self.mtp_runner.lv_graph_runner.run(
                    input_ids, positions, context
                )
                if outputs is not None:
                    if self.mtp_runner is not None:
                        self.mtp_runner.last_hidden = outputs.clone()
                    return self.model.compute_logits(outputs)
                # No graph for this bs — eager fallback
                hidden = self.model(input_ids, positions)
                if self.mtp_runner is not None:
                    self.mtp_runner.last_hidden = hidden
                return self.model.compute_logits(hidden)

            # Normal decode (seqlen_q=1)
            outputs = self.decode_graph_runner.run(input_ids, positions, context)
            if self.decode_graph_runner.returns_logits and self.mtp_runner is None:
                self._mark_fwd("model")
                self._mark_fwd("logits")
                return outputs
            if self.mtp_runner is not None:
                self.mtp_runner.last_hidden = outputs.clone()
            self._mark_fwd("model")
            logits = self.model.compute_logits(outputs)
            self._mark_fwd("logits")
            return logits

    def migrate_from_bytes(self, data: bytes) -> None:
        """Migrate using lean MigrateBatchInput bytes (no Sequence objects)."""
        self.cache_transfer.migrate_from_bytes(data=data)

    # ------------------------------------------------------------------ #
    # L3 (3FS) tiered KV cache worker RPCs (driven by collective_rpc)
    # ------------------------------------------------------------------ #
    def l3_store_blocks(self, pairs: list[tuple[int, int]]) -> int:
        """Persist (hash, block_id) GPU blocks to 3FS. Returns #stored."""
        if self.l3_store is None or not pairs:
            return 0
        return len(self.l3_store.store([tuple(p) for p in pairs]))

    def l3_load_blocks(self, pairs: list[tuple[int, int]]) -> int:
        """Load (hash, block_id) blocks from 3FS into GPU. Returns #loaded.

        Synchronizes so the data is visible before the forward pass runs.
        """
        if self.l3_store is None or not pairs:
            return 0
        n = self.l3_store.load([tuple(p) for p in pairs])
        torch.cuda.synchronize()
        return n

    def l3_stats(self) -> dict:
        return self.l3_store.stats() if self.l3_store is not None else {}

    def swap_out_blocks_to_host(
        self, tasks: list[tuple[int, list[int], list[int]]]
    ) -> int:
        """Copy GPU KV blocks into the worker-local host KV cache.

        Each task is ``(seq_id, gpu_blocks, host_blocks)``. Scheduler owns the
        table state; the worker only moves the bytes for its rank-local shard.
        """
        cache_context = get_cache_context()
        if cache_context.host_kv_cache is None:
            if tasks:
                raise RuntimeError("host KV cache is not allocated")
            return 0

        copied = 0
        for _seq_id, gpu_blocks, host_blocks in tasks:
            if len(gpu_blocks) != len(host_blocks):
                raise ValueError(
                    "swap_out_blocks_to_host requires same-size GPU/host tables"
                )
            for gpu_block, host_block in zip(gpu_blocks, host_blocks):
                self._copy_kv_block_to_host(
                    cache_context, int(gpu_block), int(host_block)
                )
                copied += 1
        if copied:
            torch.cuda.synchronize()
        return copied

    def swap_in_blocks_from_host(
        self, tasks: list[tuple[int, list[int], list[int]]]
    ) -> int:
        """Copy worker-local host KV blocks back into newly allocated GPU blocks.

        Each task is ``(seq_id, host_blocks, gpu_blocks)``.
        """
        cache_context = get_cache_context()
        if cache_context.host_kv_cache is None:
            if tasks:
                raise RuntimeError("host KV cache is not allocated")
            return 0

        copied = 0
        for _seq_id, host_blocks, gpu_blocks in tasks:
            if len(host_blocks) != len(gpu_blocks):
                raise ValueError(
                    "swap_in_blocks_from_host requires same-size host/GPU tables"
                )
            for host_block, gpu_block in zip(host_blocks, gpu_blocks):
                self._copy_kv_block_from_host(
                    cache_context, int(host_block), int(gpu_block)
                )
                copied += 1
        if copied:
            torch.cuda.synchronize()
        return copied

    @staticmethod
    def _copy_kv_block_to_host(cache_context, gpu_block: int, host_block: int) -> None:
        gpu_kv_cache = cache_context.kv_cache
        host_kv_cache = cache_context.host_kv_cache
        if cache_context.mode == "dsv4":
            host_kv_cache[:, host_block].copy_(
                gpu_kv_cache[:, gpu_block], non_blocking=True
            )
        elif cache_context.mode == "mla":
            host_kv_cache[:, :, host_block].copy_(
                gpu_kv_cache[:, :, gpu_block], non_blocking=True
            )
        else:
            host_kv_cache[:, :, host_block].copy_(
                gpu_kv_cache[:, :, gpu_block], non_blocking=True
            )

    @staticmethod
    def _copy_kv_block_from_host(
        cache_context, host_block: int, gpu_block: int
    ) -> None:
        gpu_kv_cache = cache_context.kv_cache
        host_kv_cache = cache_context.host_kv_cache
        if cache_context.mode == "dsv4":
            gpu_kv_cache[:, gpu_block].copy_(
                host_kv_cache[:, host_block], non_blocking=True
            )
        elif cache_context.mode == "mla":
            gpu_kv_cache[:, :, gpu_block].copy_(
                host_kv_cache[:, :, host_block], non_blocking=True
            )
        else:
            gpu_kv_cache[:, :, gpu_block].copy_(
                host_kv_cache[:, :, host_block], non_blocking=True
            )

    def _standard_sample(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        aux,
        num_seqs: int,
        is_prefill: bool,
    ):
        """Standard sampling path (prefill or normal decode without lazy verify).

        Returns ``(input_ids, logprobs_or_None)`` — when any seq in the
        batch has ``return_completion_logprobs=True``, ``logprobs`` is a
        ``[num_seqs]`` float32 tensor of the chosen-token logprobs;
        otherwise None and the original (compile-cached) forward path is
        used. The logprob array is TP-all-reduced via a sum (non-rank-0
        contributes zeros), mirroring the existing input_ids reduction.
        """
        tp_rank = get_dist_context().attn_tp_rank
        want_lp = bool(getattr(aux, "any_return_completion_logprobs", False))
        logprobs = None
        if tp_rank == 0:
            context = get_batch_context()
            if (
                is_prefill
                and context.sampling_seq_indices is not None
                and context.sampling_seq_indices.numel() == 0
            ):
                # Intermediate static PP microbatches update cache/state only.
                # Avoid invoking the sampler (and its compiled kernels) with a
                # zero-row logits tensor; the driver discards this placeholder.
                input_ids = input_ids.new_zeros(num_seqs)
                if want_lp:
                    logprobs = torch.zeros(
                        num_seqs, dtype=torch.float32, device=input_ids.device
                    )
                return input_ids, logprobs
            greedy_only = not want_lp and all(
                float(t) < 1e-5 for t in getattr(aux, "temperatures", ())
            )
            if greedy_only:
                if str(logits.dtype).startswith("torch.float8"):
                    logits = logits.float()
                return logits.argmax(dim=-1), None

            temperatures = prepare_sample_from_aux(aux)
            if is_prefill and context.sampling_seq_indices is not None:
                temps_filtered = temperatures[context.sampling_seq_indices]
                if want_lp:
                    sampled, lp_filtered = self.sampler.forward_with_logprobs(
                        logits, temps_filtered
                    )
                    input_ids = input_ids.new_zeros(num_seqs)
                    input_ids[context.sampling_seq_indices] = sampled
                    logprobs = torch.zeros(num_seqs, dtype=torch.float32, device="cuda")
                    logprobs[context.sampling_seq_indices] = lp_filtered.float()
                else:
                    sampled = self.sampler(logits, temps_filtered)
                    input_ids = input_ids.new_zeros(num_seqs)
                    input_ids[context.sampling_seq_indices] = sampled
            else:
                if want_lp:
                    input_ids, logprobs = self.sampler.forward_with_logprobs(
                        logits, temperatures
                    )
                    logprobs = logprobs.float()
                else:
                    input_ids = self.sampler(logits, temperatures)
        else:
            # Non-leader TP ranks don't sample; they return a placeholder. The
            # real token is distributed via the control plane, not a GPU
            # collective (see below).
            input_ids = input_ids.new_zeros([num_seqs])
            if want_lp:
                logprobs = torch.zeros(num_seqs, dtype=torch.float32, device="cuda")
        # NOTE: no cross-TP collective here (nano-vllm style). Only TP rank 0
        # samples; the engine consumes rank-0's result only
        # (executor.run(...)[::tp_size] in LLMEngine.step), and the sampled token
        # reaches the other TP ranks for the *next* step through the serialized
        # batch bytes (scheduler.postprocess appends the token to the sequences,
        # which are re-serialized and dispatched to every worker, then read back
        # via prepare_decode_bytes -> seq.last_token). Reducing/broadcasting the
        # token over MCCL is both unnecessary and deadlocks the P2P path on this
        # hardware after a few decode steps.
        return input_ids, logprobs

    @torch.inference_mode()
    def prepare_from_bytes(self, data: bytes, is_prefill: bool) -> _PreparedRun:
        """Prepare a RunnerIn payload into tensors/context for a later run."""
        _rcfg = get_runner_config()
        _timing = _rcfg.step_timing
        _timer = _StepTimer() if _timing else None
        if _timer is not None:
            _timer.mark("start")
        # CUDA-event forward breakdown: only active on the steps that will be
        # reported (interval + rank gate, using the post-increment run_count),
        # so event creation overhead is ~0 on ordinary steps.
        _fwd_timer = None
        if _timing:
            _rk = _rcfg.step_timing_rank
            if (_rk < 0 or self.rank == _rk) and (
                (self.run_count + 1) % _rcfg.step_timing_interval == 0
            ):
                _fwd_timer = _CudaForwardTimer()
        # GPU-idle probe: record a start event before any GPU op of this step.
        # Combined with the previous step's end event it yields the inter-step
        # GPU-idle gap (no host sync on the hot path; one sync only on report
        # steps, after the gap window has already closed).
        _gap_start_evt = None
        _gap_report = False
        if (
            _rcfg.gpu_idle_probe
            and torch.cuda.is_initialized()
            and (_rcfg.step_timing_rank < 0 or self.rank == _rcfg.step_timing_rank)
        ):
            _gap_start_evt = torch.cuda.Event(enable_timing=True)
            _gap_start_evt.record()
            _gap_report = (self.run_count + 1) % _rcfg.step_timing_interval == 0
        sp_rank = get_dist_context().attn_sp_rank
        runner_in = RunnerIn.from_bytes(data)
        aux = runner_in.aux(sp_rank)
        num_seqs = aux.num_group_seqs
        if _timer is not None:
            _timer.mark("rpc_in")

        is_dummy = False
        if num_seqs == 0:
            is_dummy = True
            runner_in = RunnerIn.dummy(
                self.engine_id,
                get_cache_context().num_local_kvcache_blocks,
                is_prefill,
            )
            data = runner_in.to_bytes()
            aux = runner_in.aux(sp_rank)
            num_seqs = aux.num_group_seqs

        if is_prefill and self.mtp_runner is not None:
            self.mtp_runner.reset_lazy_verify_state()

        # --- Prepare inputs ---
        has_lazy_verify = False
        if is_prefill:
            if not self.vision_manager.has_embeds:
                vision_slots = runner_in.vision_slots()
                if vision_slots:
                    self.vision_manager.fetch_rdma(
                        vision_slots, self.model.model.embed_tokens.weight.dtype
                    )
            input_ids, positions = self.input_preparer.prepare_prefill_bytes(
                data, aux, is_dummy
            )
        else:
            input_ids, positions = self.input_preparer.prepare_decode_bytes(
                data, aux, is_dummy
            )
            if _timer is not None:
                _timer.mark("prep")

            if (
                self.mtp_runner is not None
                and self.config.mode == "decode"
                and not is_dummy
            ):
                self.mtp_runner.restore_disagg_handoff(
                    aux.seq_ids, aux.state_slots, num_seqs
                )

            if (
                self.mtp_runner is not None
                and self.mtp_runner.has_drafts
                and not is_dummy
                and self.mtp_runner.can_lazy_verify(aux.seq_ids, positions, num_seqs)
            ):
                has_lazy_verify = True
                input_ids, positions = self.mtp_runner.prepare_lazy_verify_decode(
                    input_ids, positions, num_seqs
                )

        if input_ids.numel() == 0:
            logger.critical(
                "EMPTY input_ids before run_model! rank=%s is_prefill=%s "
                "is_dummy=%s input_ids.shape=%s positions.shape=%s num_seqs=%s",
                self.rank,
                is_prefill,
                is_dummy,
                input_ids.shape,
                positions.shape,
                num_seqs,
            )

        return _PreparedRun(
            input_ids=input_ids,
            positions=positions,
            aux=aux,
            num_seqs=num_seqs,
            is_prefill=is_prefill,
            is_dummy=is_dummy,
            has_lazy_verify=has_lazy_verify,
            batch_context=get_batch_context(),
            hca_tile_scheduler_metadata=get_hca_context().tile_scheduler_metadata,
            runner_config=_rcfg,
            timer=_timer,
            fwd_timer=_fwd_timer,
            gap_start_evt=_gap_start_evt,
            gap_report=_gap_report,
        )

    def _activate_prepared_context(self, prepared: _PreparedRun) -> None:
        """Restore runtime context captured by prepare_from_bytes."""
        context = prepared.batch_context
        set_batch_context(
            is_prefill=context.is_prefill,
            max_bs=context.max_bs,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            slot_mapping=context.slot_mapping,
            context_lens=context.context_lens,
            block_tables=context.block_tables,
            is_dummy=context.is_dummy,
            gdn_conv_states=context.gdn_conv_states,
            gdn_recurrent_states=context.gdn_recurrent_states,
            gdn_state_slots=context.gdn_state_slots,
            dsv4_state_slots=context.dsv4_state_slots,
            dsv4_compressed_block_tables=context.dsv4_compressed_block_tables,
            hisparse_slots=context.hisparse_slots,
            hisparse_slot_mapping=context.hisparse_slot_mapping,
            hisparse_num_real_reqs=context.hisparse_num_real_reqs,
            hisparse_phase_id=context.hisparse_phase_id,
            num_tokens_per_seq=context.num_tokens_per_seq,
            sampling_token_indices=context.sampling_token_indices,
            sampling_seq_indices=context.sampling_seq_indices,
            paged_attention_strategy=context.paged_attention_strategy,
            graph_attention_strategy=context.graph_attention_strategy,
            decode_page_plan_key=context.decode_page_plan_key,
            mtp_draft_safe=context.mtp_draft_safe,
        )
        get_hca_context().tile_scheduler_metadata = prepared.hca_tile_scheduler_metadata

    @torch.inference_mode()
    def run_prepared(self, prepared: _PreparedRun):
        """Run a previously prepared payload and build the wire result."""
        self._activate_prepared_context(prepared)
        input_ids = prepared.input_ids
        positions = prepared.positions
        aux = prepared.aux
        num_seqs = prepared.num_seqs
        is_prefill = prepared.is_prefill
        is_dummy = prepared.is_dummy
        has_lazy_verify = prepared.has_lazy_verify
        _rcfg = prepared.runner_config
        _timer = prepared.timer
        _fwd_timer = prepared.fwd_timer
        _gap_start_evt = prepared.gap_start_evt
        _gap_report = prepared.gap_report
        self._fwd_timer = _fwd_timer

        # --- Forward ---
        if _fwd_timer is not None:
            _fwd_timer.mark("fwd_start")
        target_input_ids = input_ids
        logits = self.run_model(input_ids, positions, is_prefill)
        if _timer is not None:
            _timer.mark("forward")
        if is_prefill and self.vision_manager.has_embeds:
            self.vision_manager.clear()

        # Non-final pipeline stages produce no logits/tokens; they only need to
        # have run their local layers (and pushed hidden states downstream). The
        # engine reads tokens from the last stage only, so return a benign
        # placeholder here.
        if logits is None:
            self._advance_profiler()
            reset_runtime_contexts()
            self.run_count += 1
            self._fwd_timer = None
            return [[0] for _ in range(num_seqs)]

        # --- Sampling ---
        num_accepted = None
        step_logprobs = None  # [num_seqs] float32 when shipping logprobs
        if not is_prefill and has_lazy_verify:
            num_accepted = torch.zeros(num_seqs, dtype=torch.int64, device="cuda")
            input_ids, step_logprobs = self.mtp_runner.lazy_verify_sample(
                logits, aux, num_seqs, num_accepted
            )
        else:
            input_ids, step_logprobs = self._standard_sample(
                logits, input_ids, aux, num_seqs, is_prefill
            )
        if _fwd_timer is not None:
            _fwd_timer.mark("sample")
        if _timer is not None:
            _timer.mark("sample")

        # --- MTP draft generation ---
        if self.mtp_runner is not None and is_prefill:
            # Seed the predictor cache from shifted prompt tokens and target
            # hidden states. All attention-DP ranks (including dummy ranks)
            # enter the same number of predictor MoE collectives.
            self.mtp_runner.generate_prefill_and_store(
                target_input_ids, positions, input_ids, aux, num_seqs
            )
            if self.config.mode == "prefill":
                self.mtp_runner.publish_disagg_handoff(
                    aux.seq_ids, aux.state_slots, num_seqs
                )
        elif self.mtp_runner is not None:
            # Every rank in the FFN EP group must enter the predictor
            # collectives in the same order. In attention-DP serving the
            # inactive shards receive dummy batches; skipping draft generation
            # on those ranks deadlocks the active shard inside the MTP MoE.
            # Dummy outputs remain local and are discarded by the scheduler.
            self.mtp_runner.generate_and_store(
                input_ids, positions, aux, num_seqs, has_lazy_verify, num_accepted
            )
        if _timer is not None:
            _timer.mark("mtp")

        # --- Profiler step ---
        self._advance_profiler()

        self.run_count += 1
        batch_out = get_batch_out_context()
        batch_out.token_ids.append(input_ids[None, ...])
        if step_logprobs is not None:
            # ``step_logprobs`` shape is [num_seqs] float32 (zero on
            # non-rank-0 / non-sampled seqs).
            if batch_out.step_logprobs is None:
                batch_out.step_logprobs = []
            batch_out.step_logprobs.append(step_logprobs[None, ...])

        # --- Build output ---
        # ``logprobs_per_seq`` is ``list[list[float]]`` parallel to ``result``
        # when shipping logprobs is enabled; None otherwise. Engine-server
        # serializes both into StepOut.
        logprobs_per_seq = None
        if batch_out.step_logprobs:
            logprobs_per_seq = torch.cat(batch_out.step_logprobs, dim=0).T.tolist()
        if self.mtp_runner is None and logprobs_per_seq is None:
            result = [[int(token)] for token in input_ids.tolist()]
        elif self.mtp_runner is not None:
            if logprobs_per_seq is not None:
                logprobs_per_seq = self.mtp_runner.build_output_logprobs(
                    logprobs_per_seq
                )
            result = self.mtp_runner.build_output_tokens(self.rank)
        else:
            result = torch.cat(batch_out.token_ids, dim=0).T.tolist()
        # Keep wire compat: return bare list when no logprobs were requested,
        # tuple ``(tokens, logprobs)`` when they were. Engine-side decoder
        # normalises both shapes.
        if logprobs_per_seq is not None:
            result = (result, logprobs_per_seq)
        reset_runtime_contexts()
        if _timer is not None:
            _timer.mark("tail")
            _timer.report(
                self.rank,
                self.run_count,
                num_seqs,
                is_prefill,
                _rcfg.step_timing_interval,
                _rcfg.step_timing_rank,
            )
        if _fwd_timer is not None:
            _fwd_timer.report(self.rank, self.run_count, num_seqs, is_prefill)
        self._fwd_timer = None
        # GPU-idle probe: close this step's window and, on report steps,
        # measure the gap from the previous step's GPU end to this step's
        # GPU start (idle) plus this step's GPU-busy span.
        if _gap_start_evt is not None:
            _gap_end_evt = torch.cuda.Event(enable_timing=True)
            _gap_end_evt.record()
            prev_end = getattr(self, "_gap_prev_end_evt", None)
            if _gap_report and prev_end is not None:
                _gap_end_evt.synchronize()
                gap_ms = prev_end.elapsed_time(_gap_start_evt)
                busy_ms = _gap_start_evt.elapsed_time(_gap_end_evt)
                total_ms = gap_ms + busy_ms
                duty = (busy_ms / total_ms * 100) if total_ms > 0 else 0.0
                kind = "prefill" if is_prefill else "decode"
                get_logger().info(
                    f"[gpu_idle] r{self.rank} {kind} step={self.run_count} "
                    f"num_seqs={num_seqs} gap={gap_ms:.3f}ms busy={busy_ms:.3f}ms "
                    f"step={total_ms:.3f}ms duty={duty:.0f}%"
                )
            self._gap_prev_end_evt = _gap_end_evt
        return result

    @torch.inference_mode()
    def run_from_bytes(self, data: bytes, is_prefill: bool):
        """Run model from lean RunnerIn bytes (completely Sequence-free)."""
        prepared = self.prepare_from_bytes(data, is_prefill)
        return self.run_prepared(prepared)

    def _init_graph_runners(self):
        """Initialize CUDAGraph runners for decode, MTP, and lazy verify."""
        config = self.config
        hf_config = config.hf_config
        hf_config.max_position_embeddings = max(
            config.max_model_len, hf_config.max_position_embeddings
        )
        cache_ctx = get_cache_context()

        self.decode_graph_runner = DecodeGraphRunner(config, hf_config, cache_ctx)
        graph_pool = self.decode_graph_runner.capture(self.model, cache_ctx)
        torch.cuda.synchronize()

        if self.mtp_runner is not None:
            self.mtp_runner.init_graph_runners(self.model, graph_pool, cache_ctx)
