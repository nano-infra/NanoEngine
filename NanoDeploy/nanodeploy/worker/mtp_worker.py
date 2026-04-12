"""MTP (Multi-Token Prediction) speculative decoding worker."""

from __future__ import annotations

import torch
import torch.distributed as dist

from nanodeploy.config import Config
from nanodeploy.context.cache import get_cache_context
from nanodeploy.context.context import get_context, set_context
from nanodeploy.context.distributed import get_dist_context
from nanodeploy.layers.sampler import Sampler
from nanodeploy.logging import get_logger
from nanodeploy.worker.graph_runner import LazyVerifyGraphRunner, MTPGraphRunner
from nanodeploy.worker.input_preparer import prepare_sample_from_aux

logger = get_logger("NANODEPLOY")


class MTPWorker:
    """Manages MTP speculative decoding: lazy verify, draft generation, and sampling."""

    def __init__(self, config: Config, mtp_model, sampler: Sampler):
        self.config = config
        self.mtp_model = mtp_model
        self.sampler = sampler
        self.last_hidden: torch.Tensor | None = None
        self.mtp_graph_runner: MTPGraphRunner | None = None
        self.lv_graph_runner: LazyVerifyGraphRunner | None = None
        self._prev_drafts: list[torch.Tensor] | None = None
        self._prev_sampled_token: torch.Tensor | None = None
        self._mtp_verified_tokens: torch.Tensor | None = None
        self._mtp_num_accepted: torch.Tensor | None = None

    @property
    def has_drafts(self) -> bool:
        return self._prev_drafts is not None

    def reset_lazy_verify_state(self):
        self._prev_drafts = None
        self._prev_sampled_token = None

    def init_graph_runners(self, target_model, graph_pool, cache_ctx):
        """Create and capture MTP + lazy-verify CUDAGraph runners."""
        config = self.config
        hf_config = config.hf_config

        self.mtp_graph_runner = MTPGraphRunner(config, hf_config)
        self.mtp_graph_runner.capture(self.mtp_model, graph_pool)

        self.lv_graph_runner = LazyVerifyGraphRunner(config, hf_config, cache_ctx)
        self.lv_graph_runner.capture(target_model, graph_pool, cache_ctx)

    def cleanup(self):
        """Delete graph runners to free CUDA resources."""
        if self.lv_graph_runner is not None:
            del self.lv_graph_runner
            self.lv_graph_runner = None
        if self.mtp_graph_runner is not None:
            del self.mtp_graph_runner
            self.mtp_graph_runner = None

    # ------------------------------------------------------------------
    # Lazy verify: input expansion
    # ------------------------------------------------------------------

    def prepare_lazy_verify_decode(
        self, input_ids: torch.Tensor, positions: torch.Tensor, num_seqs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand decode from seqlen_q=1 to seqlen_q=2 for lazy verification.

        Transforms:
            input_ids:  [bs] → [bs*2] interleaved [prev_sampled_0, draft_0, ...]
            positions:  [bs] → [bs*2] interleaved [pos_0, pos_0+1, ...]

        Also updates context: slot_mapping, context_lens, num_tokens_per_seq,
        and MLA metadata.
        """
        sp_rank = get_dist_context().attn_sp_rank
        block_size = self.config.kvcache_block_size
        context = get_context()

        prev_sampled = self._prev_sampled_token  # [bs] token from prev step
        prev_draft = self._prev_drafts[0]  # [bs] draft from prev step

        # Build interleaved input_ids
        new_input_ids = torch.empty(
            num_seqs * 2, dtype=input_ids.dtype, device=input_ids.device
        )
        new_input_ids[0::2] = prev_sampled[:num_seqs]
        new_input_ids[1::2] = prev_draft[:num_seqs]

        # Build interleaved positions
        new_positions = torch.empty(
            num_seqs * 2, dtype=positions.dtype, device=positions.device
        )
        new_positions[0::2] = positions[:num_seqs]
        new_positions[1::2] = positions[:num_seqs] + 1

        # Expand slot_mapping: 2 slots per seq
        old_ctx = context.context_lens[sp_rank, :num_seqs]
        new_slot_mapping = torch.empty(
            num_seqs * 2, dtype=torch.int32, device=old_ctx.device
        )

        for offset in range(2):
            pos = old_ctx - 1 + offset
            blk_idx = (pos // block_size).long()
            off_in_block = (pos % block_size).int()
            row_idx = torch.arange(num_seqs, device=pos.device)
            max_blocks = context.block_tables.shape[2]
            blk_idx = torch.clamp(blk_idx, max=max_blocks - 1)
            page_ids = context.block_tables[sp_rank, row_idx, blk_idx]
            new_slot_mapping[offset::2] = (page_ids * block_size + off_in_block).int()

        # +1 for the draft token
        context.context_lens[sp_rank, :num_seqs] += 1

        # Recompute MLA metadata for seqlen_q=2
        hf_config = self.config.hf_config
        is_mla = getattr(hf_config, "kv_lora_rank", 0) > 0
        if is_mla:
            import flash_mla

            new_tile_sched, _ = flash_mla.get_mla_metadata()
        else:
            new_tile_sched = context.tile_scheduler_metadata

        # Update context for seqlen_q=2
        context.slot_mapping = new_slot_mapping
        context.num_tokens_per_seq = 2
        if is_mla:
            context.tile_scheduler_metadata = new_tile_sched

        return new_input_ids, new_positions

    # ------------------------------------------------------------------
    # Lazy verify: sampling + rollback
    # ------------------------------------------------------------------

    def lazy_verify_sample(
        self,
        logits: torch.Tensor,
        aux,
        num_seqs: int,
        num_accepted: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from lazy-verify logits and rollback GDN states for rejected seqs.

        Args:
            logits: [bs*2, vocab] interleaved verify/bonus logits.
            aux: BatchAuxData with temperatures.
            num_seqs: number of sequences.
            num_accepted: output tensor [num_seqs] filled with acceptance counts.

        Returns:
            input_ids: [num_seqs] newly sampled token per sequence.
        """
        verify_logits = logits[0::2]  # [bs, vocab]
        bonus_logits = logits[1::2]  # [bs, vocab]

        verified_tokens = torch.zeros(1, num_seqs, dtype=torch.int64, device="cuda")
        tp_rank = get_dist_context().attn_tp_rank

        if tp_rank == 0:
            temperatures = prepare_sample_from_aux(aux)
            target_pred = self.sampler(verify_logits, temperatures)
            bonus_pred = self.sampler(bonus_logits, temperatures)
            prev_draft_0 = self._prev_drafts[0]

            accepted_mask = target_pred == prev_draft_0
            verified_tokens[0] = prev_draft_0
            num_accepted.copy_(accepted_mask.long())
            input_ids = torch.where(accepted_mask, bonus_pred, target_pred)
        else:
            input_ids = torch.zeros(num_seqs, dtype=torch.int64, device="cuda")

        dist.all_reduce(input_ids, group=get_dist_context().attn_tp_group)
        dist.all_reduce(verified_tokens, group=get_dist_context().attn_tp_group)
        dist.all_reduce(num_accepted, group=get_dist_context().attn_tp_group)

        self._mtp_verified_tokens = verified_tokens
        self._mtp_num_accepted = num_accepted

        # Rollback context_lens and GDN states for rejected sequences
        context = get_context()
        sp_rank = get_dist_context().attn_sp_rank
        rejected_mask = num_accepted == 0
        if rejected_mask.any():
            context.context_lens[sp_rank, :num_seqs] -= rejected_mask.int()

            _cache_ctx = get_cache_context()
            if _cache_ctx.gdn_conv_states is not None:
                active_slots = context.gdn_state_slots[:num_seqs]
                real_slot_mask = active_slots < _cache_ctx.gdn_max_active_slots
                rollback_mask = rejected_mask & real_slot_mask
                if rollback_mask.any():
                    backup_offset = _cache_ctx.gdn_max_active_slots
                    rej_active = active_slots[rollback_mask]
                    rej_backup = rej_active + backup_offset
                    _cache_ctx.gdn_conv_states[:, rej_active] = (
                        _cache_ctx.gdn_conv_states[:, rej_backup]
                    )
                    _cache_ctx.gdn_recurrent_states[:, rej_active] = (
                        _cache_ctx.gdn_recurrent_states[:, rej_backup]
                    )

        return input_ids

    # ------------------------------------------------------------------
    # Draft generation + context save/restore
    # ------------------------------------------------------------------

    def generate_and_store(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        aux,
        num_seqs: int,
        has_lazy_verify: bool,
        num_accepted: torch.Tensor | None,
    ) -> None:
        """Generate MTP drafts and save state for next-step lazy verification.

        Saves and restores the decode context around MTP forward passes.
        """
        decode_context = get_context()
        saved_token_ids = decode_context.token_ids

        tp_rank = get_dist_context().attn_tp_rank
        temperatures = prepare_sample_from_aux(aux) if tp_rank == 0 else None

        # Select correct hidden states and positions for MTP input
        if has_lazy_verify:
            h_verify = self.last_hidden[0::2]  # [bs, hidden_size]
            h_bonus = self.last_hidden[1::2]  # [bs, hidden_size]
            accepted_mask = num_accepted > 0
            mtp_hidden = torch.where(accepted_mask.unsqueeze(-1), h_bonus, h_verify)
            mtp_positions = positions[0::2] + 1 + num_accepted
        else:
            mtp_hidden = self.last_hidden
            mtp_positions = positions

        drafts = self._generate_mtp_drafts(
            input_ids, mtp_positions, mtp_hidden, temperatures, num_seqs
        )

        self._prev_drafts = drafts
        self._prev_sampled_token = input_ids.clone()

        # Restore decode context (MTP draft gen overwrites it)
        set_context(
            is_prefill=decode_context.is_prefill,
            max_bs=decode_context.max_bs,
            slot_mapping=decode_context.slot_mapping,
            context_lens=decode_context.context_lens,
            block_tables=decode_context.block_tables,
            is_dummy=decode_context.is_dummy,
            tile_scheduler_metadata=decode_context.tile_scheduler_metadata,
            gdn_conv_states=decode_context.gdn_conv_states,
            gdn_recurrent_states=decode_context.gdn_recurrent_states,
            gdn_state_slots=decode_context.gdn_state_slots,
        )
        get_context().token_ids = saved_token_ids

    # ------------------------------------------------------------------
    # Output assembly
    # ------------------------------------------------------------------

    def build_output_tokens(self, rank: int) -> list[list[int]]:
        """Assemble final output tokens, interleaving verified MTP drafts."""
        if self._mtp_verified_tokens is not None:
            base = torch.cat(get_context().token_ids, dim=0)  # [loop_count, num_seqs]
            verified = self._mtp_verified_tokens  # [N, num_seqs]
            accepted = self._mtp_num_accepted  # [num_seqs]
            result = []
            for s in range(base.shape[1]):
                n_acc = int(accepted[s].item())
                seq_tokens = []
                for j in range(n_acc):
                    seq_tokens.append(int(verified[j, s].item()))
                for t in range(base.shape[0]):
                    seq_tokens.append(int(base[t, s].item()))
                result.append(seq_tokens)
            if rank == 0:
                logger.debug(f"OUTPUT tokens[0]: {result[0]}")
            self._mtp_verified_tokens = None
            self._mtp_num_accepted = None
            return result
        else:
            return torch.cat(get_context().token_ids, dim=0).T.tolist()

    # ------------------------------------------------------------------
    # Internal: MTP forward helpers
    # ------------------------------------------------------------------

    def _set_mtp_context(self, num_seqs: int):
        """Set context for MTP forward (prefill mode, seq_len=1, no KV cache)."""
        cu_seqlens = torch.arange(num_seqs + 1, dtype=torch.int32, device="cuda")
        set_context(
            is_prefill=True,
            max_bs=self.config.max_num_seqs,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=1,
            max_seqlen_k=1,
            slot_mapping=None,
            block_tables=None,
            is_dummy=False,
            use_low_latency_ep=True,
        )

    def _run_mtp_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        spec_step_idx: int,
        bs: int,
    ) -> torch.Tensor:
        """Run one MTP speculative step. Uses CUDAGraph if captured."""
        if self.mtp_graph_runner is not None:
            output = self.mtp_graph_runner.run(input_ids, positions, hidden_states, bs)
            if output is not None:
                return output

        # Eager fallback
        self._set_mtp_context(bs)
        return self.mtp_model(
            input_ids,
            positions,
            hidden_states,
            spec_step_idx=spec_step_idx,
        )

    @torch.inference_mode()
    def _generate_mtp_drafts(
        self,
        sampled_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        temperatures: torch.Tensor | None,
        num_seqs: int,
    ) -> list[torch.Tensor]:
        """Generate draft tokens using MTP layers (decode only)."""
        drafts = []
        current_hidden = hidden_states
        current_ids = sampled_ids
        current_pos = positions + 1

        for step in range(self.config.num_speculative_tokens):
            mtp_hidden = self._run_mtp_step(
                current_ids, current_pos, current_hidden, step, num_seqs
            )

            # Set MTP context for compute_logits (ParallelLMHead reads is_prefill)
            self._set_mtp_context(num_seqs)
            mtp_logits = self.mtp_model.compute_logits(mtp_hidden, spec_step_idx=step)

            # Sample on tp_rank 0, broadcast
            tp_rank = get_dist_context().attn_tp_rank
            if tp_rank == 0:
                draft_ids = self.sampler(mtp_logits, temperatures)
            else:
                draft_ids = current_ids.new_zeros(num_seqs)
            dist.all_reduce(draft_ids, group=get_dist_context().attn_tp_group)

            drafts.append(draft_ids)
            current_hidden = mtp_hidden
            current_ids = draft_ids
            current_pos = current_pos + 1

        return drafts
