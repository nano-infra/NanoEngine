"""Input preparation: prefill/decode bytes → GPU tensors + context setup."""

from __future__ import annotations

import torch

from nanodeploy._cpp import prepare_decode_from_bytes, prepare_prefill_from_bytes
from nanodeploy.config import Config
from nanodeploy.context.cache import get_cache_context
from nanodeploy.context.context import get_context, set_context
from nanodeploy.context.distributed import get_dist_context
from nanodeploy.logging import get_logger

logger = get_logger("NANODEPLOY")


def prepare_sample_from_aux(aux) -> torch.Tensor:
    """Build temperature tensor from BatchAuxData."""
    return torch.tensor(aux.temperatures, dtype=torch.float32, pin_memory=True).cuda(
        non_blocking=True
    )


class InputPreparer:
    """Prepares prefill/decode input tensors and context from serialized bytes."""

    def __init__(self, config: Config):
        self.config = config

    def prepare_prefill_bytes(self, data: bytes, aux, is_dummy: bool = False):
        sp_rank = get_dist_context().attn_sp_rank
        sp_size = get_dist_context().attn_sp_world_size
        block_size = self.config.kvcache_block_size

        meta = prepare_prefill_from_bytes(
            data,
            sp_rank,
            sp_size,
            block_size,
            self.config.max_num_seqs,
            self.config.num_kvcache_blocks,
        )

        if len(meta.input_ids) == 0:
            logger.critical(
                "prepare_prefill_from_bytes returned empty input_ids! "
                "is_dummy=%s block_size=%s max_num_seqs=%s",
                is_dummy,
                block_size,
                self.config.max_num_seqs,
            )

        input_ids = torch.tensor(
            meta.input_ids, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        positions = torch.tensor(
            meta.positions, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(
            meta.cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            meta.cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            meta.slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        block_tables = None
        if meta.use_block_tables:
            block_tables = (
                torch.tensor(meta.block_tables_flat, dtype=torch.int32, pin_memory=True)
                .reshape(sp_size, self.config.max_num_seqs, meta.max_num_blocks)
                .cuda(non_blocking=True)
            )

        cache_ctx = get_cache_context()
        gdn_state_slots = None
        if cache_ctx.gdn_conv_states is not None:
            dummy_gdn_slot = cache_ctx.gdn_conv_states.shape[1] - 1
            gdn_state_slots = torch.tensor(
                [
                    s if 0 <= s < dummy_gdn_slot else dummy_gdn_slot
                    for s in aux.state_slots
                ],
                dtype=torch.int64,
                pin_memory=True,
            ).cuda(non_blocking=True)

        # Chunked prefill: selective lm_head — only compute logits for final-chunk seqs.
        sampling_token_indices = None
        sampling_seq_indices = None
        num_seqs = aux.num_group_seqs
        if len(meta.sampling_token_indices) < num_seqs:
            sampling_token_indices = torch.tensor(
                meta.sampling_token_indices, dtype=torch.int64, pin_memory=True
            ).cuda(non_blocking=True)
            sampling_seq_indices = torch.tensor(
                meta.sampling_seq_indices, dtype=torch.int64, pin_memory=True
            ).cuda(non_blocking=True)

        set_context(
            is_prefill=True,
            max_bs=self.config.max_num_seqs,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=meta.max_seqlen_q,
            max_seqlen_k=meta.max_seqlen_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            is_dummy=is_dummy,
            gdn_conv_states=cache_ctx.gdn_conv_states,
            gdn_recurrent_states=cache_ctx.gdn_recurrent_states,
            gdn_state_slots=gdn_state_slots,
            sampling_token_indices=sampling_token_indices,
            sampling_seq_indices=sampling_seq_indices,
        )
        return input_ids, positions

    def prepare_decode_bytes(self, data: bytes, aux, is_dummy: bool = False):
        sp_rank = get_dist_context().attn_sp_rank
        sp_size = get_dist_context().attn_sp_world_size
        block_size = self.config.kvcache_block_size

        try:
            meta = prepare_decode_from_bytes(
                data,
                sp_rank,
                sp_size,
                block_size,
                self.config.max_num_seqs,
                self.config.num_kvcache_blocks,
            )
        except (IndexError, ValueError, RuntimeError) as e:
            logger.error(
                "prepare_decode_from_bytes failed: %s (block_size=%s max_num_seqs=%s)",
                str(e),
                block_size,
                self.config.max_num_seqs,
            )
            raise

        input_ids = torch.tensor(
            meta.input_ids, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        positions = torch.tensor(
            meta.positions, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            meta.slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        context_lens = (
            torch.tensor(meta.context_lens_flat, dtype=torch.int32, pin_memory=True)
            .reshape(sp_size, self.config.max_num_seqs)
            .cuda(non_blocking=True)
        )

        if len(meta.block_tables_flat) == 0:
            block_tables = torch.empty((1, 0, 0), dtype=torch.int32).cuda(
                non_blocking=True
            )
        else:
            block_tables = (
                torch.tensor(meta.block_tables_flat, dtype=torch.int32, pin_memory=True)
                .reshape(sp_size, -1, meta.max_num_blocks)
                .cuda(non_blocking=True)
            )

        config = self.config
        hf_config = config.hf_config
        is_mla = getattr(hf_config, "kv_lora_rank", 0) > 0
        if is_mla:
            import flash_mla

            mla_num_kv_heads = 1
            context_lens_for_mla = context_lens[sp_rank, : aux.num_group_seqs]
            new_tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()
        else:
            new_tile_scheduler_metadata = None

        cache_ctx = get_cache_context()
        gdn_state_slots = None
        if cache_ctx.gdn_conv_states is not None:
            dummy_gdn_slot = cache_ctx.gdn_conv_states.shape[1] - 1
            gdn_state_slots = torch.tensor(
                [
                    s if 0 <= s < dummy_gdn_slot else dummy_gdn_slot
                    for s in aux.state_slots
                ],
                dtype=torch.int64,
                pin_memory=True,
            ).cuda(non_blocking=True)

        set_context(
            is_prefill=False,
            max_bs=self.config.max_num_seqs,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            is_dummy=is_dummy,
            tile_scheduler_metadata=new_tile_scheduler_metadata,
            gdn_conv_states=cache_ctx.gdn_conv_states,
            gdn_recurrent_states=cache_ctx.gdn_recurrent_states,
            gdn_state_slots=gdn_state_slots,
        )

        return input_ids, positions

    def update_decode_inplace(
        self, input_ids: torch.Tensor, positions: torch.Tensor, num_seqs: int
    ):
        """Update decode metadata in-place for multi-step decode (no Sequence needed)."""
        positions.add_(1)
        block_size = self.config.kvcache_block_size
        context = get_context()

        sp_rank = get_dist_context().attn_sp_rank

        # Update context length (now reflects the NEW token count)
        context.context_lens[sp_rank, :num_seqs].add_(1)

        # Recalculate slot_mapping from context_lens and block_tables.
        new_ctx = context.context_lens[sp_rank, :num_seqs]  # already incremented
        block_idx = (new_ctx - 1) // block_size  # which block the new token falls in
        offset_in_block = (new_ctx - 1) % block_size  # offset within that block
        row_indices = torch.arange(num_seqs, device=block_idx.device)
        page_ids = context.block_tables[sp_rank, row_indices, block_idx.long()]
        context.slot_mapping[:num_seqs] = page_ids * block_size + offset_in_block

        return input_ids, positions
