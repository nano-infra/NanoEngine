"""CUDAGraph capture and replay management for NanoDeploy model runner.

Isolates all graph buffer allocation, capture, and replay logic from
ModelRunner, keeping the runner focused on orchestration.

Provides:
- DecodeGraphRunner:      standard decode (seqlen_q=1) graphs
- LazyVerifyGraphRunner:  lazy verify decode (seqlen_q=2) graphs
- MTPGraphRunner:         MTP draft forward graphs
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from nanodeploy.context.context import Context, reset_context, set_context
from nanodeploy.context.distributed import get_dist_context
from nanodeploy.context.expert_context import ExpertContext
from nanodeploy.logging import get_logger

logger = get_logger("NANODEPLOY")


def _make_bs_list(max_bs: int) -> list[int]:
    """Build a sorted list of batch sizes to capture: small powers-of-2 + multiples of 16."""
    return [x for x in [1, 2, 4, 8] if x <= max_bs] + list(range(16, max_bs + 1, 16))


# ---------------------------------------------------------------------------
# Decode (seqlen_q = 1)
# ---------------------------------------------------------------------------


class DecodeGraphRunner:
    """CUDAGraph capture / replay for standard decode (one token per seq)."""

    def __init__(self, config, hf_config, cache_ctx):
        max_bs = min(config.max_num_seqs, 512)
        block_size = cache_ctx.block_size
        max_num_blocks = (config.max_model_len + block_size - 1) // block_size
        is_mla = getattr(hf_config, "kv_lora_rank", 0) > 0

        # Persistent input / output buffers
        self._input_ids = torch.zeros(max_bs, dtype=torch.int64)
        self._positions = torch.zeros(max_bs, dtype=torch.int64)
        self._slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        self._context_lens = torch.zeros(1, max_bs, dtype=torch.int32)
        self._block_tables = torch.zeros(1, max_bs, max_num_blocks, dtype=torch.int32)
        self._outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # MLA-specific
        if is_mla:
            import flash_mla

            mla_kv = 1
            self._tile_sched, self._num_splits = flash_mla.get_mla_metadata(
                torch.ones(max_bs, dtype=torch.int32, device="cuda"),
                hf_config.num_attention_heads // mla_kv,
                mla_kv,
            )
        else:
            self._tile_sched = self._num_splits = None

        # GDN state slots
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
        self._graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._graph_map: dict[int, list[int]] = {}
        self._graph_pool = None

    # -- public properties --------------------------------------------------

    @property
    def graph_pool(self):
        return self._graph_pool

    @property
    def bs_list(self):
        return self._bs_list

    # -- capture ------------------------------------------------------------

    @torch.inference_mode()
    def capture(self, model, cache_ctx):
        """Capture CUDAGraphs for every batch size.

        Returns:
            The ``torch.cuda.graphs.MemPool`` for sharing with other runners.
        """
        logger.info("Capturing decode CUDAGraphs...")

        for master_bs in reversed(self._bs_list):
            attn_bs = master_bs
            self._graph_map[master_bs] = [attn_bs]

            logger.info(f"Capturing graph - (master_bs={master_bs}, attn_bs={attn_bs})")

            set_context(
                is_prefill=False,
                max_bs=self._max_num_seqs,
                slot_mapping=self._slot_mapping[:master_bs],
                context_lens=self._context_lens,
                block_tables=self._block_tables,
                tile_scheduler_metadata=self._tile_sched,
                num_splits=self._num_splits,
                gdn_conv_states=cache_ctx.gdn_conv_states,
                gdn_recurrent_states=cache_ctx.gdn_recurrent_states,
                gdn_state_slots=(
                    self._gdn_state_slots[:master_bs]
                    if self._gdn_state_slots is not None
                    else None
                ),
            )

            # Warmup
            self._outputs[:master_bs] = model(
                self._input_ids[:master_bs], self._positions[:master_bs]
            )

            # Capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, self._graph_pool):
                self._outputs[:master_bs] = model(
                    self._input_ids[:master_bs], self._positions[:master_bs]
                )

            if self._graph_pool is None:
                self._graph_pool = graph.pool()

            self._graphs[(master_bs, attn_bs)] = graph

            torch.cuda.synchronize()
            dist.barrier(group=get_dist_context().cuda_world_group)
            reset_context()

        logger.info(f"Finished capturing {len(self._graphs)} decode CUDAGraphs")
        return self._graph_pool

    # -- replay -------------------------------------------------------------

    def run(
        self, input_ids: torch.Tensor, positions: torch.Tensor, context: Context
    ) -> torch.Tensor:
        """Copy inputs into captured buffers, select graph, replay."""
        bs = input_ids.size(0)
        master_bs = next(x for x in self._bs_list if x >= bs)

        attn_bs = bs
        valid = self._graph_map.get(master_bs)
        if valid is None:
            raise RuntimeError(f"No graph map for master_bs={master_bs}")
        attn_bs = next(x for x in valid if x >= attn_bs)

        # Copy inputs
        self._input_ids[:bs] = input_ids
        self._positions[:bs] = positions
        self._slot_mapping.fill_(-1)
        self._slot_mapping[:bs] = context.slot_mapping
        self._context_lens.zero_()
        self._context_lens[:, : context.context_lens.shape[1]].copy_(
            context.context_lens
        )
        self._block_tables.zero_()
        self._block_tables[
            :, : context.block_tables.size(1), : context.block_tables.size(2)
        ] = context.block_tables

        if self._is_mla:
            self._tile_sched.zero_()
            self._tile_sched.copy_(context.tile_scheduler_metadata)
            self._num_splits.zero_()
            self._num_splits[: context.num_splits.shape[0]].copy_(context.num_splits)

        if self._gdn_state_slots is not None:
            self._gdn_state_slots.fill_(self._dummy_gdn_slot)
            if context.gdn_state_slots is not None:
                self._gdn_state_slots[:bs].copy_(context.gdn_state_slots)

        self._graphs[(master_bs, attn_bs)].replay()
        return self._outputs[:bs]


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

        if is_mla:
            import flash_mla

            mla_kv = 1
            # seqlen_q=2 → double the q-heads-per-kv ratio
            self._tile_sched, self._num_splits = flash_mla.get_mla_metadata(
                torch.ones(max_bs, dtype=torch.int32, device="cuda"),
                2 * hf_config.num_attention_heads // mla_kv,
                mla_kv,
            )
        else:
            self._tile_sched = self._num_splits = None

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

            set_context(
                is_prefill=False,
                max_bs=self._max_num_seqs,
                slot_mapping=self._slot_mapping[:n_tokens],
                context_lens=self._context_lens,
                block_tables=self._block_tables,
                is_dummy=False,
                tile_scheduler_metadata=self._tile_sched,
                num_splits=self._num_splits,
                num_tokens_per_seq=2,
                gdn_conv_states=cache_ctx.gdn_conv_states,
                gdn_recurrent_states=cache_ctx.gdn_recurrent_states,
                gdn_state_slots=(
                    self._gdn_state_slots[:bs]
                    if self._gdn_state_slots is not None
                    else None
                ),
            )

            # Warmup
            self._outputs[:n_tokens] = model(
                self._input_ids[:n_tokens], self._positions[:n_tokens]
            )

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, graph_pool):
                self._outputs[:n_tokens] = model(
                    self._input_ids[:n_tokens], self._positions[:n_tokens]
                )

            self._graphs[bs] = graph
            reset_context()

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
            self._tile_sched.zero_()
            self._tile_sched.copy_(context.tile_scheduler_metadata)
            self._num_splits.zero_()
            self._num_splits[: context.num_splits.shape[0]].copy_(context.num_splits)

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

            set_context(
                is_prefill=True,
                max_bs=self._max_num_seqs,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=1,
                max_seqlen_k=1,
                slot_mapping=None,
                block_tables=None,
                is_dummy=False,
                use_low_latency_ep=True,
            )

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

        reset_context()
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
