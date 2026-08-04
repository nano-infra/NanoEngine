"""CUDAGraph capture and replay management for DLEngine model runner.

Isolates all graph buffer allocation, capture, and replay logic from
ModelRunner, keeping the runner focused on orchestration.

Provides:
- DecodeGraphRunner:      standard decode (seqlen_q=1) graphs
- LazyVerifyGraphRunner:  lazy verify decode (seqlen_q=2) graphs
- MTPGraphRunner:         MTP draft forward graphs
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from dlengine.context_v2.batch import Context, get_batch_context, set_batch_context
from dlengine.context_v2.cache.hca import get_hca_context
from dlengine.context_v2.cache.hisparse import (
    build_hot_slot_mapping,
    get_hisparse_context,
)
from dlengine.context_v2.cache.mla import get_mla_context
from dlengine.context_v2.distributed import get_dist_context
from dlengine.context_v2.expert import ExpertContext, set_expert_context
from dlengine.context_v2.graph import (
    DecodeGraphContext,
    FlashInferDecodeGraphConfig,
    get_graph_context,
    PagedAttentionStrategy,
)
from dlengine.context_v2.management import reset_runtime_contexts
from dlengine.logging import get_logger

logger = get_logger("DLENGINE")


def _flashinfer_decode_enabled() -> bool:
    return os.environ.get("DLENGINE_USE_FLASHINFER_DECODE", "1") == "1"


def _env_flag(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0") == "1"


def _capture_logits_enabled() -> bool:
    return os.environ.get("DLENGINE_DECODE_GRAPH_CAPTURE_LOGITS", "1") == "1"


def _reuse_flashinfer_plan_enabled() -> bool:
    return os.environ.get("DLENGINE_FLASHINFER_REUSE_PAGE_PLAN", "1") == "1"


def _make_bs_list(max_bs: int) -> list[int]:
    """Build a sorted list of batch sizes to capture: small powers-of-2 + multiples of 16."""
    return [x for x in [1, 2, 4, 8] if x <= max_bs] + list(range(16, max_bs + 1, 16))


# ---------------------------------------------------------------------------
# Decode (seqlen_q = 1)
# ---------------------------------------------------------------------------


class DecodeGraphRunner:
    """CUDAGraph capture / replay for standard decode (one token per seq)."""

    def __init__(self, config, hf_config, cache_ctx):
        self.config = config
        max_bs = min(config.max_num_seqs, 512)
        block_size = cache_ctx.block_size
        max_num_blocks = (config.max_model_len + block_size - 1) // block_size
        is_mla = getattr(hf_config, "kv_lora_rank", 0) > 0
        is_dsv4 = hf_config.architectures[0] == "DeepseekV4ForCausalLM"

        # Persistent input / output buffers
        self._input_ids = torch.zeros(max_bs, dtype=torch.int64)
        self._positions = torch.zeros(max_bs, dtype=torch.int64)
        self._slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        self._context_lens = torch.zeros(1, max_bs, dtype=torch.int32)
        self._block_tables = torch.zeros(1, max_bs, max_num_blocks, dtype=torch.int32)
        self._outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self._returns_logits = (
            _capture_logits_enabled() and get_dist_context().attn_tp_world_size == 1
        )
        logits_dtype = getattr(hf_config, "dtype", torch.get_default_dtype())
        if getattr(logits_dtype, "is_floating_point", False) and str(
            logits_dtype
        ).startswith("torch.float8"):
            logits_dtype = torch.bfloat16
        self._logits = (
            torch.zeros(max_bs, hf_config.vocab_size, dtype=logits_dtype)
            if self._returns_logits
            else None
        )
        self._hisparse_slots = torch.full(
            (max_bs,), config.max_num_seqs, dtype=torch.int64
        )
        self._hisparse_slot_mapping = torch.full((max_bs,), -1, dtype=torch.int32)
        hidden_size_per_layer_input = getattr(
            hf_config, "hidden_size_per_layer_input", 0
        )
        self._per_layer_token_part = (
            torch.empty(
                max_bs,
                hf_config.num_hidden_layers,
                hidden_size_per_layer_input,
                dtype=getattr(hf_config, "dtype", torch.get_default_dtype()),
            )
            if hidden_size_per_layer_input
            else None
        )

        # MLA-specific: per-BS FlashMLASchedMeta created during capture
        if is_mla or is_dsv4:
            import flash_mla

            self._flash_mla = flash_mla
        self._sched_metas: dict[int, object] = {}
        self._sparse_sched_metas: dict[int, object] = {}
        self._uses_flash_mla_sched = (
            (is_mla or is_dsv4) and torch.cuda.get_device_capability()[0] < 10
        )

        # NSA indexer detection (V3.2): sparse decode needs its own FlashMLASchedMeta
        self._has_indexer = is_mla and getattr(hf_config, "index_n_heads", 0) > 0

        # GDN state slots
        self._gdn_state_slots = None
        self._dummy_gdn_slot = None
        if cache_ctx.gdn_conv_states is not None:
            self._dummy_gdn_slot = cache_ctx.gdn_conv_states.shape[1] - 1
            self._gdn_state_slots = torch.full(
                (max_bs,), self._dummy_gdn_slot, dtype=torch.int64
            )

        # DSv4 compressor state slots (parallel to gdn_state_slots)
        self._dsv4_state_slots = None
        self._dummy_dsv4_slot = None
        if is_dsv4:
            # Dummy slot = max_num_seqs (compressor buffer has max+1 slots).
            self._dummy_dsv4_slot = config.max_num_seqs
            self._dsv4_state_slots = torch.full(
                (max_bs,), self._dummy_dsv4_slot, dtype=torch.int64
            )

        # DSv4 compressed-cache block tables (per ratio).  Persistent buffers
        # indexed by state_slot (matches dsv4_state_slots' addressing) — shape
        # [max_bs+1, max_blocks_per_seq], with row max_bs reserved as the dummy.
        # Filled with dummy_page; per-replay copy_() overwrites with InputPreparer's
        # state-slot-indexed table (also [max_bs+1, max_blocks]).
        self._dsv4_compressed_block_tables: dict[int, torch.Tensor] = {}
        self._dsv4_compressed_dummy_pages: dict[int, int] = {}
        if is_dsv4:
            pool_cfg = getattr(cache_ctx, "dsv4_compressed_pool_config", {}) or {}
            dummies = getattr(cache_ctx, "dsv4_compressed_dummy_page", {}) or {}
            for ratio, (num_pages, _page_size, max_blocks) in pool_cfg.items():
                if max_blocks <= 0:
                    continue
                dummy_page = dummies.get(ratio, num_pages)
                self._dsv4_compressed_dummy_pages[ratio] = dummy_page
                self._dsv4_compressed_block_tables[ratio] = torch.full(
                    (max_bs + 1, max_blocks), dummy_page, dtype=torch.int32
                )

        self._is_mla = is_mla
        self._is_dsv4 = is_dsv4
        self._max_num_seqs = config.max_num_seqs
        self._block_size = block_size
        self._num_heads = getattr(hf_config, "num_attention_heads")
        # Graph paged-decode metadata must describe the allocated cache, not
        # necessarily the model's default attention shape. Gemma4, for example,
        # keeps 256-dim sliding K/V in HiSparse while its paged global K/V cache
        # uses global_head_dim=512.
        self._num_kv_heads = cache_ctx.num_local_kv_heads
        self._head_dim = cache_ctx.head_dim
        self._softmax_scale = (
            1.0
            if hf_config.architectures[0]
            in ("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration")
            else self._head_dim**-0.5
        )
        self._decode_dtype = cache_ctx.kv_cache.dtype
        self._flashinfer_decode_enabled = (
            _flashinfer_decode_enabled() and not is_mla and not is_dsv4
        )
        self._flashinfer_use_tensor_cores = _env_flag(
            "DLENGINE_FLASHINFER_GRAPH_TENSOR_CORES", False
        )
        self._flashinfer_disable_split_kv = _env_flag(
            "DLENGINE_FLASHINFER_GRAPH_DISABLE_SPLIT_KV", True
        )
        self._flashinfer_fixed_split_size = max(
            0, int(os.environ.get("DLENGINE_FLASHINFER_GRAPH_FIXED_SPLIT_SIZE", "0"))
        )
        self._flashinfer_reuse_page_plan = _reuse_flashinfer_plan_enabled()
        self._graph_ctx = get_graph_context()

        self._bs_list = _make_bs_list(max_bs)
        self._graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._graph_map: dict[int, list[int]] = {}
        self._decode_ctx = self._graph_ctx.set_decode(
            DecodeGraphContext(
                max_num_seqs=self._max_num_seqs,
                block_size=self._block_size,
                max_num_blocks=max_num_blocks,
                is_mla=self._is_mla,
                is_dsv4=self._is_dsv4,
                has_indexer=self._has_indexer,
                input_ids=self._input_ids,
                positions=self._positions,
                slot_mapping=self._slot_mapping,
                context_lens=self._context_lens,
                block_tables=self._block_tables,
                outputs=self._outputs,
                returns_logits=self._returns_logits,
                logits=self._logits,
                hisparse_slots=self._hisparse_slots,
                hisparse_slot_mapping=self._hisparse_slot_mapping,
                per_layer_token_part=self._per_layer_token_part,
                gdn_state_slots=self._gdn_state_slots,
                dummy_gdn_slot=self._dummy_gdn_slot,
                dsv4_state_slots=self._dsv4_state_slots,
                dummy_dsv4_slot=self._dummy_dsv4_slot,
                dsv4_compressed_block_tables=self._dsv4_compressed_block_tables,
                dsv4_compressed_dummy_pages=self._dsv4_compressed_dummy_pages,
                sched_metas=self._sched_metas,
                sparse_sched_metas=self._sparse_sched_metas,
                graphs=self._graphs,
                graph_map=self._graph_map,
            )
        )
        self._graph_ctx.set_flashinfer_decode(
            FlashInferDecodeGraphConfig(
                enabled=self._flashinfer_decode_enabled,
                max_num_blocks=max_num_blocks,
                block_size=block_size,
                num_heads=self._num_heads,
                num_kv_heads=self._num_kv_heads,
                head_dim=self._head_dim,
                softmax_scale=self._softmax_scale,
                dtype=self._decode_dtype,
                use_tensor_cores=self._flashinfer_use_tensor_cores,
                disable_split_kv=self._flashinfer_disable_split_kv,
                fixed_split_size=self._flashinfer_fixed_split_size,
                reuse_page_plan=self._flashinfer_reuse_page_plan,
            )
            if self._flashinfer_decode_enabled
            else None
        )

    # -- public properties --------------------------------------------------

    @property
    def graph_pool(self):
        return self._decode_ctx.graph_pool

    @property
    def bs_list(self):
        return self._bs_list

    @property
    def returns_logits(self) -> bool:
        return self._decode_ctx.returns_logits

    def _ensure_flashinfer_graph_wrapper(self, master_bs: int):
        if not self._flashinfer_decode_enabled:
            return None
        state = self._graph_ctx.flashinfer_decode
        return state.ensure_wrapper(master_bs) if state is not None else None

    def _plan_flashinfer_graph_wrapper(
        self,
        master_bs: int,
        bs: int,
        page_plan_key: tuple[int, ...] | None = None,
    ) -> object | None:
        if not self._flashinfer_decode_enabled:
            return None
        state = self._graph_ctx.flashinfer_decode
        if state is None:
            return None
        g = self._decode_ctx
        return state.plan(
            master_bs,
            bs,
            g.block_tables[0, :bs],
            g.context_lens[0, :bs],
            page_plan_key=page_plan_key,
        )

    # -- capture ------------------------------------------------------------

    @torch.inference_mode()
    def capture(self, model, cache_ctx):
        """Capture CUDAGraphs for every batch size.

        Returns:
            The ``torch.cuda.graphs.MemPool`` for sharing with other runners.
        """
        logger.info("Capturing decode CUDAGraphs...")
        self._model = model
        g = self._decode_ctx

        for master_bs in reversed(self._bs_list):
            attn_bs = master_bs
            g.graph_map[master_bs] = [attn_bs]

            logger.info(f"Capturing graph - (master_bs={master_bs}, attn_bs={attn_bs})")

            # Each BS gets its own FlashMLASchedMeta (kernel validates batch size)
            sched_meta = None
            sparse_sched_meta = None
            if self._uses_flash_mla_sched:
                sched_meta, _ = self._flash_mla.get_mla_metadata()
                g.sched_metas[master_bs] = sched_meta
            if self._has_indexer and self._uses_flash_mla_sched:
                sparse_sched_meta, _ = self._flash_mla.get_mla_metadata()
                g.sparse_sched_metas[master_bs] = sparse_sched_meta
                # Indexer needs non-zero context_lens during warmup so that
                # deep_gemm MQA-logits and topk operate on valid data.
                g.context_lens[:, :master_bs] = 1
            if self._flashinfer_decode_enabled:
                g.context_lens[:, :master_bs] = 1

            hisparse_ctx = get_hisparse_context()
            if hisparse_ctx.num_real_reqs is not None:
                hisparse_ctx.num_real_reqs.fill_(master_bs)

            context = set_batch_context(
                is_prefill=False,
                max_bs=self._max_num_seqs,
                slot_mapping=g.slot_mapping[:master_bs],
                context_lens=g.context_lens,
                block_tables=g.block_tables,
                gdn_conv_states=cache_ctx.gdn_conv_states,
                gdn_recurrent_states=cache_ctx.gdn_recurrent_states,
                gdn_state_slots=(
                    g.gdn_state_slots[:master_bs]
                    if g.gdn_state_slots is not None
                    else None
                ),
                dsv4_state_slots=(
                    g.dsv4_state_slots[:master_bs]
                    if g.dsv4_state_slots is not None
                    else None
                ),
                # Pass FULL [max_bs+1, max_blocks] table — indexed by state_slot
                # (which can be up to max_bs = max_num_seqs for the dummy).
                dsv4_compressed_block_tables=(
                    dict(g.dsv4_compressed_block_tables)
                    if g.dsv4_compressed_block_tables
                    else None
                ),
                hisparse_slots=g.hisparse_slots[:master_bs],
                hisparse_slot_mapping=(
                    build_hot_slot_mapping(
                        g.hisparse_slots[:master_bs],
                        g.positions[:master_bs],
                        g.hisparse_slot_mapping[:master_bs],
                    )
                    if getattr(self.config, "cache_plan", None) is not None
                    and self.config.cache_plan.has_gqa()
                    else g.slot_mapping[:master_bs]
                ),
                hisparse_num_real_reqs=get_hisparse_context().num_real_reqs,
                graph_attention_strategy=(
                    PagedAttentionStrategy.FLASHINFER
                    if self._flashinfer_decode_enabled
                    else None
                ),
            )
            get_hca_context().tile_scheduler_metadata = sched_meta
            get_mla_context().sparse_tile_scheduler_metadata = sparse_sched_meta
            flashinfer_wrapper = self._plan_flashinfer_graph_wrapper(
                master_bs, master_bs
            )
            self._graph_ctx.set_active_flashinfer_decode_wrapper(flashinfer_wrapper)

            model_kwargs = {}
            if g.per_layer_token_part is not None:
                token_part = model.model.prepare_decode_graph_token_part(
                    g.input_ids[:master_bs],
                    g.per_layer_token_part.device,
                    g.per_layer_token_part.dtype,
                )
                g.per_layer_token_part[:master_bs].copy_(token_part)
                model_kwargs["per_layer_token_part"] = g.per_layer_token_part[
                    :master_bs
                ]
            # Warmup
            hidden = model(
                g.input_ids[:master_bs], g.positions[:master_bs], **model_kwargs
            )
            if g.returns_logits:
                g.logits[:master_bs] = model.compute_logits(hidden)
            else:
                g.outputs[:master_bs] = hidden

            # Create a fresh FlashMLASchedMeta so that the scheduling
            # metadata kernel (get_decoding_sched_meta) is captured in the
            # graph.  dense_decode_fwd only launches it when the metadata
            # tensor is None; after warmup it is non-None, so without this
            # reset the graph would replay with stale scheduling data.
            if self._uses_flash_mla_sched:
                sched_meta, _ = self._flash_mla.get_mla_metadata()
                g.sched_metas[master_bs] = sched_meta
                get_hca_context().tile_scheduler_metadata = sched_meta
            if self._is_dsv4:
                # DSv4 uses per-layer sched_metas (mixed compress_ratio configs).
                # Drop the warmup-initialized metas so the capture pass creates
                # fresh ones — this ensures the scheduling kernel runs inside
                # the graph rather than during eager warmup.
                from dlengine.models.deepseek_v4.deepseek_v4 import DeepseekV4Attention

                for m in model.modules():
                    if isinstance(m, DeepseekV4Attention):
                        m._dsv4_sched_metas.pop(master_bs, None)
            if self._has_indexer and self._uses_flash_mla_sched:
                sparse_sched_meta, _ = self._flash_mla.get_mla_metadata()
                g.sparse_sched_metas[master_bs] = sparse_sched_meta
                get_mla_context().sparse_tile_scheduler_metadata = sparse_sched_meta
            flashinfer_wrapper = self._plan_flashinfer_graph_wrapper(
                master_bs, master_bs
            )
            self._graph_ctx.set_active_flashinfer_decode_wrapper(flashinfer_wrapper)

            # Capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, g.graph_pool):
                hidden = model(
                    g.input_ids[:master_bs], g.positions[:master_bs], **model_kwargs
                )
                if g.returns_logits:
                    g.logits[:master_bs] = model.compute_logits(hidden)
                else:
                    g.outputs[:master_bs] = hidden

            if g.graph_pool is None:
                g.graph_pool = graph.pool()

            g.graphs[(master_bs, attn_bs)] = graph

            torch.cuda.synchronize()
            dist.barrier(group=get_dist_context().cuda_world_group)
            self._graph_ctx.clear_step_context()
            reset_runtime_contexts()

        logger.info(f"Finished capturing {len(g.graphs)} decode CUDAGraphs")
        return g.graph_pool

    # -- replay -------------------------------------------------------------

    def run(
        self, input_ids: torch.Tensor, positions: torch.Tensor, context: Context
    ) -> torch.Tensor:
        """Copy inputs into captured buffers, select graph, replay."""
        g = self._decode_ctx
        bs = input_ids.size(0)
        master_bs = next(x for x in self._bs_list if x >= bs)

        attn_bs = bs
        valid = g.graph_map.get(master_bs)
        if valid is None:
            raise RuntimeError(f"No graph map for master_bs={master_bs}")
        attn_bs = next(x for x in valid if x >= attn_bs)

        # Copy inputs
        g.input_ids[:bs] = input_ids
        g.positions[:bs] = positions
        if g.per_layer_token_part is not None:
            token_part = self._model.model.prepare_decode_graph_token_part(
                input_ids,
                g.per_layer_token_part.device,
                g.per_layer_token_part.dtype,
            )
            g.per_layer_token_part[:bs].copy_(token_part)
        g.slot_mapping.fill_(-1)
        g.slot_mapping[:bs] = context.slot_mapping
        g.hisparse_slots.fill_(g.max_num_seqs)
        if context.hisparse_slots is not None:
            g.hisparse_slots[:bs].copy_(context.hisparse_slots[:bs])
        hisparse_ctx = get_hisparse_context()
        if hisparse_ctx.num_real_reqs is not None:
            hisparse_ctx.num_real_reqs.fill_(bs)
        if context.hisparse_slot_mapping is not None:
            build_hot_slot_mapping(
                g.hisparse_slots[:master_bs],
                g.positions[:master_bs],
                g.hisparse_slot_mapping[:master_bs],
            )
        g.context_lens.zero_()
        g.context_lens[:, : context.context_lens.shape[1]].copy_(context.context_lens)
        g.block_tables.zero_()
        g.block_tables[
            :, : context.block_tables.size(1), : context.block_tables.size(2)
        ] = context.block_tables
        if self._flashinfer_decode_enabled and bs < master_bs:
            g.context_lens[:, bs:master_bs] = 1

        if self._is_mla:
            pass  # FlashMLASchedMeta is managed internally by the kernel; no copy needed

        if g.gdn_state_slots is not None:
            g.gdn_state_slots.fill_(g.dummy_gdn_slot)
            if context.gdn_state_slots is not None:
                g.gdn_state_slots[:bs].copy_(context.gdn_state_slots)

        if g.dsv4_state_slots is not None:
            g.dsv4_state_slots.fill_(g.dummy_dsv4_slot)
            if context.dsv4_state_slots is not None:
                g.dsv4_state_slots[:bs].copy_(context.dsv4_state_slots)

        if g.dsv4_compressed_block_tables:
            for ratio, persistent in g.dsv4_compressed_block_tables.items():
                persistent.fill_(g.dsv4_compressed_dummy_pages[ratio])
                src = (context.dsv4_compressed_block_tables or {}).get(ratio)
                if src is not None:
                    # Source is also [max_bs+1, max_blocks] (state-slot indexed).
                    # Both shapes match so we can copy directly. If they differ
                    # (e.g., src is shorter), copy the available prefix.
                    n = min(src.shape[0], persistent.shape[0])
                    persistent[:n].copy_(src[:n])

        page_plan_key = context.decode_page_plan_key
        if page_plan_key is not None:
            page_plan_key = page_plan_key[:bs] + (1,) * max(0, master_bs - bs)

        self._plan_flashinfer_graph_wrapper(
            master_bs, master_bs, page_plan_key=page_plan_key
        )
        g.graphs[(master_bs, attn_bs)].replay()
        if g.returns_logits:
            return g.logits[:bs]
        return g.outputs[:bs]


# ---------------------------------------------------------------------------
# Lazy Verify (seqlen_q = 2)
# ---------------------------------------------------------------------------


class LazyVerifyGraphRunner:
    """CUDAGraph capture / replay for lazy verify decode (two tokens per seq)."""

    def __init__(self, config, hf_config, cache_ctx):
        max_bs = min(config.max_num_seqs, 512)
        block_size = cache_ctx.block_size
        max_num_blocks = (config.max_model_len + block_size - 1) // block_size
        is_mla = getattr(hf_config, "kv_lora_rank", 0) > 0

        # Buffers sized for max_bs seqs × 2 tokens
        self._input_ids = torch.zeros(max_bs * 2, dtype=torch.int64)
        self._positions = torch.zeros(max_bs * 2, dtype=torch.int64)
        self._slot_mapping = torch.full((max_bs * 2,), -1, dtype=torch.int32)
        self._context_lens = torch.zeros(1, max_bs, dtype=torch.int32)
        self._block_tables = torch.zeros(1, max_bs, max_num_blocks, dtype=torch.int32)
        self._outputs = torch.zeros(max_bs * 2, hf_config.hidden_size)

        # MLA-specific: per-BS FlashMLASchedMeta created during capture
        if is_mla:
            import flash_mla

            self._flash_mla = flash_mla
        self._sched_metas: dict[int, object] = {}
        self._sparse_sched_metas: dict[int, object] = {}
        self._uses_flash_mla_sched = (
            is_mla and torch.cuda.get_device_capability()[0] < 10
        )

        # NSA indexer detection (V3.2): sparse decode needs its own FlashMLASchedMeta
        self._has_indexer = is_mla and getattr(hf_config, "index_n_heads", 0) > 0

        self._gdn_state_slots = None
        self._dummy_gdn_slot = None
        if cache_ctx.gdn_conv_states is not None:
            self._dummy_gdn_slot = cache_ctx.gdn_conv_states.shape[1] - 1
            self._gdn_state_slots = torch.full(
                (max_bs,), self._dummy_gdn_slot, dtype=torch.int64
            )

        self._is_mla = is_mla
        self._max_num_seqs = config.max_num_seqs

        self._bs_list = _make_bs_list(max_bs)
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}

    # -- capture ------------------------------------------------------------

    @torch.inference_mode()
    def capture(self, model, graph_pool, cache_ctx):
        """Capture lazy-verify CUDAGraphs using the shared pool."""
        logger.info("Capturing lazy verify CUDAGraphs (seqlen_q=2)...")

        for bs in reversed(self._bs_list):
            n_tokens = bs * 2
            logger.info(f"Capturing lazy verify graph - bs={bs} (n_tokens={n_tokens})")

            # Each BS gets its own FlashMLASchedMeta (kernel validates batch size)
            sched_meta = None
            sparse_sched_meta = None
            if self._uses_flash_mla_sched:
                sched_meta, _ = self._flash_mla.get_mla_metadata()
                self._sched_metas[bs] = sched_meta
            if self._has_indexer and self._uses_flash_mla_sched:
                sparse_sched_meta, _ = self._flash_mla.get_mla_metadata()
                self._sparse_sched_metas[bs] = sparse_sched_meta
                self._context_lens[:, :bs] = 1

            set_batch_context(
                is_prefill=False,
                max_bs=self._max_num_seqs,
                slot_mapping=self._slot_mapping[:n_tokens],
                context_lens=self._context_lens,
                block_tables=self._block_tables,
                is_dummy=False,
                num_tokens_per_seq=2,
                gdn_conv_states=cache_ctx.gdn_conv_states,
                gdn_recurrent_states=cache_ctx.gdn_recurrent_states,
                gdn_state_slots=(
                    self._gdn_state_slots[:bs]
                    if self._gdn_state_slots is not None
                    else None
                ),
            )
            get_hca_context().tile_scheduler_metadata = sched_meta
            get_mla_context().sparse_tile_scheduler_metadata = sparse_sched_meta

            # Warmup
            self._outputs[:n_tokens] = model(
                self._input_ids[:n_tokens], self._positions[:n_tokens]
            )

            # Fresh FlashMLASchedMeta so scheduling kernel is captured
            # (see DecodeGraphRunner.capture for full explanation).
            if self._uses_flash_mla_sched:
                sched_meta, _ = self._flash_mla.get_mla_metadata()
                self._sched_metas[bs] = sched_meta
                get_hca_context().tile_scheduler_metadata = sched_meta
            if self._has_indexer and self._uses_flash_mla_sched:
                sparse_sched_meta, _ = self._flash_mla.get_mla_metadata()
                self._sparse_sched_metas[bs] = sparse_sched_meta
                get_mla_context().sparse_tile_scheduler_metadata = sparse_sched_meta

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, graph_pool):
                self._outputs[:n_tokens] = model(
                    self._input_ids[:n_tokens], self._positions[:n_tokens]
                )

            self._graphs[bs] = graph
            reset_runtime_contexts()

        logger.info(f"Finished capturing {len(self._graphs)} lazy verify CUDAGraphs")

    # -- replay -------------------------------------------------------------

    def run(
        self, input_ids: torch.Tensor, positions: torch.Tensor, context: Context
    ) -> torch.Tensor | None:
        """Copy inputs, replay.  Returns ``None`` if no graph matches bs."""
        n_tokens = input_ids.size(0)
        bs = n_tokens // 2
        master_bs = next((x for x in self._bs_list if x >= bs), None)
        if master_bs is None or master_bs not in self._graphs:
            return None

        n = bs * 2
        self._input_ids[:n] = input_ids
        self._positions[:n] = positions
        self._slot_mapping.fill_(-1)
        self._slot_mapping[:n] = context.slot_mapping
        self._context_lens.zero_()
        self._context_lens[:, : context.context_lens.shape[1]].copy_(
            context.context_lens
        )
        self._block_tables.zero_()
        self._block_tables[
            :, : context.block_tables.size(1), : context.block_tables.size(2)
        ] = context.block_tables

        if self._is_mla:
            pass  # FlashMLASchedMeta is managed internally by the kernel; no copy needed

        if self._gdn_state_slots is not None:
            self._gdn_state_slots.fill_(self._dummy_gdn_slot)
            if context.gdn_state_slots is not None:
                self._gdn_state_slots[:bs].copy_(context.gdn_state_slots)

        self._graphs[master_bs].replay()
        return self._outputs[:n]


# ---------------------------------------------------------------------------
# MTP Draft Forward
# ---------------------------------------------------------------------------


class MTPGraphRunner:
    """CUDAGraph capture / replay for MTP speculative draft forward."""

    def __init__(self, config, hf_config):
        max_bs = min(config.max_num_seqs, 512)

        self._input_ids = torch.zeros(max_bs, dtype=torch.int64)
        self._positions = torch.zeros(max_bs, dtype=torch.int64)
        self._hidden_states = torch.zeros(max_bs, hf_config.hidden_size)
        self._outputs = torch.zeros(max_bs, hf_config.hidden_size)

        self._max_num_seqs = config.max_num_seqs
        self._bs_list = _make_bs_list(max_bs)
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        # Each bs gets its own cu_seqlens (must outlive graph lifetime)
        self._cu_seqlens_per_bs: dict[int, torch.Tensor] = {}

    # -- capture ------------------------------------------------------------

    @torch.inference_mode()
    def capture(self, mtp_model, graph_pool):
        """Capture MTP CUDAGraphs using the shared pool."""
        logger.info("Capturing MTP CUDAGraphs...")

        # Must enter low-latency EP before MTP forward
        ExpertContext.get_instance().transition_to_low_latency()

        for bs in reversed(self._bs_list):
            logger.info(f"Capturing MTP graph - bs={bs}")

            cu_seqlens = torch.arange(bs + 1, dtype=torch.int32, device="cuda")
            self._cu_seqlens_per_bs[bs] = cu_seqlens

            set_batch_context(
                is_prefill=True,
                max_bs=self._max_num_seqs,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=1,
                max_seqlen_k=1,
                slot_mapping=None,
                block_tables=None,
                is_dummy=False,
            )
            set_expert_context(use_low_latency_ep=True)

            # Warmup
            self._outputs[:bs] = mtp_model(
                self._input_ids[:bs],
                self._positions[:bs],
                self._hidden_states[:bs],
            )

            # Capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, graph_pool):
                self._outputs[:bs] = mtp_model(
                    self._input_ids[:bs],
                    self._positions[:bs],
                    self._hidden_states[:bs],
                )

            self._graphs[bs] = graph

        reset_runtime_contexts()
        logger.info(f"Finished capturing {len(self._graphs)} MTP CUDAGraphs")

    # -- replay -------------------------------------------------------------

    def run(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        bs: int,
    ) -> torch.Tensor | None:
        """Copy inputs, replay.  Returns ``None`` if no graph matches bs."""
        master_bs = next((x for x in self._bs_list if x >= bs), None)
        if master_bs is None or master_bs not in self._graphs:
            return None

        self._input_ids[:bs] = input_ids
        self._positions[:bs] = positions
        self._hidden_states[:bs] = hidden_states
        self._graphs[master_bs].replay()
        return self._outputs[:bs]
