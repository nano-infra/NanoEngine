"""Linear model-native MTP speculative decoding worker."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.distributed as dist
from dlengine.config import Config
from dlengine.logging import get_logger
from dlengine.runtime.context.batch import get_batch_context, set_batch_context
from dlengine.runtime.context.batch_out import get_batch_out_context
from dlengine.runtime.context.cache import get_cache_context
from dlengine.runtime.context.cache.hca import get_hca_context
from dlengine.runtime.context.cache.mla import get_mla_context
from dlengine.runtime.context.distributed import get_dist_context
from dlengine.runtime.context.expert import set_expert_context
from dlengine.runtime.context.graph import PagedAttentionStrategy
from dlengine.runtime.layers.sampler import Sampler
from dlengine.runtime.models.deepseek_v2.deepseek_v2 import _IndexerTopKState
from dlengine.runtime.runner.graph_runner import (
    CachedMTPChainGraphRunner,
    LazyVerifyGraphRunner,
    MTPGraphRunner,
)
from dlengine.runtime.runner.input_preparer import prepare_sample_from_aux

logger = get_logger("DLENGINE")


def _nonempty_ragged_bounds(
    cu_seqlens: torch.Tensor, num_seqs: int
) -> list[int] | None:
    """Return host ragged bounds only when every requested segment is nonempty."""
    if cu_seqlens.numel() < num_seqs + 1:
        return None
    bounds = [int(value) for value in cu_seqlens[: num_seqs + 1].tolist()]
    if any(end <= start for start, end in zip(bounds, bounds[1:])):
        return None
    return bounds


def _active_ragged_last_rows(cu_seqlens: torch.Tensor, num_seqs: int) -> torch.Tensor:
    """Return last-row indices without consuming padded cu-seqlens entries."""
    return cu_seqlens[1 : num_seqs + 1].to(torch.long) - 1


def _select_ragged_rows(
    cu_seqlens: torch.Tensor,
    row_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened token indices and compact cu-seqlens for selected rows.

    Sampling row indices are produced in scheduler order. Keeping that order
    lets a mixed PP prefill batch seed MTP only for completed requests without
    synchronizing every ragged boundary back to the host.
    """
    if cu_seqlens.numel() <= 1 or row_indices.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=cu_seqlens.device)
        zero = torch.zeros(1, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        return empty, zero

    row_indices = row_indices.to(device=cu_seqlens.device, dtype=torch.long)
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    row_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=cu_seqlens.device), lengths
    )
    selected_mask = torch.zeros(
        lengths.numel(), dtype=torch.bool, device=cu_seqlens.device
    )
    selected_mask[row_indices] = True
    token_indices = torch.nonzero(selected_mask[row_ids], as_tuple=False).flatten()
    selected_lengths = lengths.index_select(0, row_indices)
    compact_cu = torch.cat((cu_seqlens.new_zeros(1), selected_lengths.cumsum(0)))
    return token_indices, compact_cu


def _localize_packed_topk(
    logical_indices: torch.Tensor, packed_k_starts: torch.Tensor
) -> torch.Tensor:
    """Convert packed-ragged K offsets to positions local to each sequence."""
    if logical_indices.shape[0] != packed_k_starts.numel():
        raise ValueError("one packed K start is required per selected TopK row")
    offsets = packed_k_starts.to(
        device=logical_indices.device, dtype=logical_indices.dtype
    ).unsqueeze(1)
    return torch.where(logical_indices >= 0, logical_indices - offsets, -1)


def _sample_rows(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample rows with the same temperature semantics as ``Sampler``."""
    greedy = temperatures < 1e-5
    safe_temperatures = torch.where(greedy, torch.ones_like(temperatures), temperatures)
    log_probs = torch.log_softmax(
        logits.float() / safe_temperatures.unsqueeze(-1), dim=-1
    )
    noise = torch.empty_like(log_probs).exponential_(1, generator=generator)
    sampled = (log_probs - noise.clamp_min_(1e-10).log()).argmax(dim=-1)
    return torch.where(greedy, logits.argmax(dim=-1), sampled)


def linear_rejection_sample(
    logits: torch.Tensor,
    drafts: torch.Tensor,
    temperatures: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verify a greedy linear draft with exact target-distribution sampling.

    ``logits`` is sequence-major ``[B, N+1, V]`` and ``drafts`` is ``[B, N]``.
    The draft distribution is one-hot. At stochastic temperatures draft ``d``
    is therefore accepted with probability ``p(d)``; on rejection the recovery
    token is sampled from ``p`` conditioned on not being ``d``. If every draft
    is accepted, the final target row supplies the bonus token.

    Returns the next token, accepted count, original-target logprob of that next
    token, and original-target logprobs for every draft position. The latter is
    intentionally unmasked: completion logprobs describe the target policy,
    not the residual rejection distribution.
    """
    if logits.ndim != 3 or drafts.ndim != 2:
        raise ValueError("expected logits [B, N+1, V] and drafts [B, N]")
    batch_size, verify_width, _ = logits.shape
    num_drafts = drafts.shape[1]
    if drafts.shape[0] != batch_size or verify_width != num_drafts + 1:
        raise ValueError("linear verify width must equal num_drafts + 1")
    if temperatures.numel() != batch_size:
        raise ValueError("one temperature is required per sequence")

    temperatures = temperatures.to(device=logits.device, dtype=torch.float32)
    greedy = temperatures < 1e-5
    safe_temperatures = torch.where(greedy, torch.ones_like(temperatures), temperatures)
    target_logprobs = torch.log_softmax(
        logits.float() / safe_temperatures[:, None, None], dim=-1
    )

    draft_logprobs = target_logprobs[:, :num_drafts].gather(2, drafts.unsqueeze(-1))[
        ..., 0
    ]
    target_argmax = logits[:, :num_drafts].argmax(dim=-1)
    coins = torch.rand(
        (batch_size, num_drafts),
        device=logits.device,
        dtype=torch.float32,
        generator=generator,
    )
    accepts = torch.where(
        greedy[:, None],
        target_argmax == drafts,
        coins < draft_logprobs.exp(),
    )

    # A line accepts only the contiguous prefix before the first rejection.
    # Keeping this entirely on device is important: the previous Python loop's
    # ``Tensor.any()`` branches synchronized the CPU once or twice per draft
    # position and left ~0.8 ms gaps between recurrent MTP forwards.
    accepted_prefix = accepts.to(torch.int32).cumprod(dim=1)
    accepted = accepted_prefix.sum(dim=1, dtype=torch.int64)

    rows = torch.arange(batch_size, device=logits.device)
    selected_logits = logits[rows, accepted].clone()
    all_accepted = accepted == num_drafts
    rejected_draft = drafts[rows, accepted.clamp_max(num_drafts - 1)]
    selected_logits.scatter_(
        1,
        rejected_draft[:, None],
        torch.where(
            all_accepted[:, None],
            selected_logits.gather(1, rejected_draft[:, None]),
            selected_logits.new_full((batch_size, 1), float("-inf")),
        ),
    )
    next_tokens = _sample_rows(selected_logits, temperatures, generator)
    next_logprobs = target_logprobs[rows, accepted].gather(1, next_tokens[:, None])[
        :, 0
    ]

    return next_tokens, accepted, next_logprobs, draft_logprobs


def linear_greedy_verify(
    logits: torch.Tensor,
    drafts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify a top-1 line without constructing probability tensors."""
    if logits.ndim != 3 or drafts.ndim != 2:
        raise ValueError("expected logits [B, N+1, V] and drafts [B, N]")
    batch_size, verify_width, _ = logits.shape
    num_drafts = drafts.shape[1]
    if drafts.shape[0] != batch_size or verify_width != num_drafts + 1:
        raise ValueError("linear verify width must equal num_drafts + 1")

    target_tokens = logits.argmax(dim=-1)
    accepted_prefix = (
        (target_tokens[:, :num_drafts] == drafts).to(torch.int32).cumprod(dim=1)
    )
    accepted = accepted_prefix.sum(dim=1, dtype=torch.int64)
    next_tokens = target_tokens.gather(1, accepted[:, None])[:, 0]
    return next_tokens, accepted


class MTPRunner:
    """Manage recurrent linear MTP drafting and target verification."""

    def __init__(self, config: Config, mtp_model, sampler: Sampler):
        self.config = config
        self.mtp_model = mtp_model
        self.sampler = sampler
        self.last_hidden: torch.Tensor | None = None
        self.mtp_graph_runner: MTPGraphRunner | None = None
        self.cached_mtp_graph_runner: CachedMTPChainGraphRunner | None = None
        self.lv_graph_runner: LazyVerifyGraphRunner | None = None
        self._prev_seq_ids: tuple[int, ...] | None = None
        self._prev_drafts: torch.Tensor | None = None  # [B, N]
        self._selected_prev_drafts: torch.Tensor | None = None
        self._mtp_verified_tokens: torch.Tensor | None = None  # [N, B]
        self._mtp_verified_logprobs: torch.Tensor | None = None  # [N, B]
        self._mtp_num_accepted: torch.Tensor | None = None
        self._share_mtp_indexer = bool(
            getattr(config.hf_config, "index_share_for_mtp_iteration", False)
            and getattr(config.hf_config, "index_topk", None) is not None
        )

    @property
    def has_drafts(self) -> bool:
        return self._prev_drafts is not None and self._prev_seq_ids is not None

    @property
    def verify_width(self) -> int:
        return self.config.num_speculative_tokens + 1

    def reset_lazy_verify_state(self):
        self._prev_seq_ids = None
        self._prev_drafts = None
        self._selected_prev_drafts = None
        self._mtp_verified_tokens = None
        self._mtp_verified_logprobs = None
        self._mtp_num_accepted = None

    def publish_disagg_handoff(
        self, seq_ids: list[int], state_slots: list[int], num_seqs: int
    ) -> int:
        """Publish recurrent drafts into scheduler-slot-indexed PD rows.

        Valid rows are invalidated first, so a failed or zero-length prefill
        cannot expose stale drafts when a slot or sequence ID is reused.
        """
        cache_context = get_cache_context()
        handoff = cache_context.mtp_handoff
        if handoff is None or num_seqs <= 0:
            return 0

        entries = [
            (idx, int(seq_ids[idx]), int(state_slots[idx]))
            for idx in range(min(num_seqs, len(seq_ids), len(state_slots)))
            if 0 <= int(state_slots[idx]) < handoff.shape[0]
        ]
        if not entries:
            return 0
        slot_tensor = torch.tensor(
            [slot for _, _, slot in entries],
            dtype=torch.long,
            device=handoff.device,
        )
        handoff.index_fill_(0, slot_tensor, -1)

        if (
            not self.has_drafts
            or self._prev_drafts.ndim != 2
            or self._prev_drafts.shape[1] != cache_context.mtp_num_drafts
        ):
            return 0
        previous = {seq_id: idx for idx, seq_id in enumerate(self._prev_seq_ids)}
        resolved = [
            (idx, seq_id, slot, previous[seq_id])
            for idx, seq_id, slot in entries
            if seq_id in previous
        ]
        if not resolved:
            return 0

        draft_indices = torch.tensor(
            [row for _, _, _, row in resolved],
            dtype=torch.long,
            device=self._prev_drafts.device,
        )
        drafts = self._prev_drafts.index_select(0, draft_indices).to(
            device=handoff.device, dtype=handoff.dtype
        )
        resolved_slots = torch.tensor(
            [slot for _, _, slot, _ in resolved],
            dtype=torch.long,
            device=handoff.device,
        )
        handoff_rows = torch.cat(
            (
                torch.tensor(
                    [seq_id for _, seq_id, _, _ in resolved],
                    dtype=handoff.dtype,
                    device=handoff.device,
                )[:, None],
                drafts,
            ),
            dim=1,
        )
        handoff.index_copy_(0, resolved_slots, handoff_rows)
        return len(resolved)

    def restore_disagg_handoff(
        self, seq_ids: list[int], state_slots: list[int], num_seqs: int
    ) -> bool:
        """Merge migrated drafts with current in-process lazy-verify state.

        The tiny GPU-to-host validity check only runs for sequences missing
        from the previous local batch. Steady-state decode remains unchanged.
        """
        current_ids = tuple(int(seq_id) for seq_id in seq_ids[:num_seqs])
        if len(current_ids) != num_seqs or num_seqs <= 0:
            return False

        previous = {}
        if (
            self.has_drafts
            and self._prev_drafts.ndim == 2
            and self._prev_drafts.shape[1] == self.config.num_speculative_tokens
        ):
            previous = {seq_id: idx for idx, seq_id in enumerate(self._prev_seq_ids)}
        missing_positions = [
            idx for idx, seq_id in enumerate(current_ids) if seq_id not in previous
        ]
        if not missing_positions:
            return False

        cache_context = get_cache_context()
        handoff = cache_context.mtp_handoff
        if (
            handoff is None
            or handoff.shape[1] != self.config.num_speculative_tokens + 1
            or len(state_slots) < num_seqs
        ):
            return False
        missing_slots = [int(state_slots[idx]) for idx in missing_positions]
        if any(slot < 0 or slot >= handoff.shape[0] for slot in missing_slots):
            return False

        slot_tensor = torch.tensor(
            missing_slots, dtype=torch.long, device=handoff.device
        )
        rows = handoff.index_select(0, slot_tensor)
        expected_ids = torch.tensor(
            [current_ids[idx] for idx in missing_positions],
            dtype=handoff.dtype,
            device=handoff.device,
        )
        valid = rows[:, 0].eq(expected_ids) & rows[:, 1:].ge(0).all(dim=1)
        if not bool(valid.all().item()):
            logger.warning(
                "Ignoring invalid MTP PD handoff rows: seq_ids=%s slots=%s",
                [current_ids[idx] for idx in missing_positions],
                missing_slots,
            )
            return False

        migrated = {
            position: rows[row_idx, 1:]
            for row_idx, position in enumerate(missing_positions)
        }
        resolved = []
        for position, seq_id in enumerate(current_ids):
            if position in migrated:
                resolved.append(migrated[position])
            else:
                resolved.append(self._prev_drafts[previous[seq_id]])
        self._prev_seq_ids = current_ids
        self._prev_drafts = torch.stack(resolved, dim=0)
        self._selected_prev_drafts = None
        # Mark the local copy consumed. Remote prefill rows are overwritten on
        # slot reuse; the seq-id guard protects both sides from stale state.
        handoff[slot_tensor, 0] = -1
        logger.info(
            "Restored MTP PD handoff: seq_ids=%s slots=%s",
            [current_ids[idx] for idx in missing_positions],
            missing_slots,
        )
        return True

    def _new_mtp_indexer_state(self) -> _IndexerTopKState | None:
        return _IndexerTopKState() if self._share_mtp_indexer else None

    def _select_mtp_indexer_seed(
        self,
        state: _IndexerTopKState | None,
        rows: torch.Tensor,
        source_context,
        num_seqs: int,
        *,
        packed_k_starts: torch.Tensor | None = None,
    ) -> _IndexerTopKState | None:
        """Select and physicalize the DSA row that seeded the first draft."""
        if state is None or state.logical_indices is None:
            return state
        selected = state.select_rows(rows)
        if selected.physical_indices is None:
            from dlengine.runtime.layers.hopper.attention import (
                topk_indices_to_physical,
            )

            sp_rank = get_dist_context().attn_sp_rank
            block_tables = source_context.block_tables
            if block_tables is None:
                raise RuntimeError("MTP DSA IndexShare requires paged block tables")
            if packed_k_starts is not None:
                # Sparse prefill addresses a concatenated ragged K tensor, but
                # page tables are sequence-local. Without removing each
                # sequence's packed offset, sequence 2+ can gather past its
                # allocated page-table width on a cold multi-request batch.
                selected.logical_indices = _localize_packed_topk(
                    selected.logical_indices, packed_k_starts
                )
            selected.physical_indices = topk_indices_to_physical(
                selected.logical_indices,
                block_tables[sp_rank, :num_seqs],
                self.config.kvcache_block_size,
            )
        return selected

    def can_lazy_verify(
        self, seq_ids: list[int], positions: torch.Tensor, num_seqs: int
    ) -> bool:
        """Return whether stored drafts can safely verify this decode batch."""
        if not self.has_drafts or len(seq_ids) < num_seqs:
            return False
        previous = {seq_id: idx for idx, seq_id in enumerate(self._prev_seq_ids)}
        selected = [previous.get(int(seq_id)) for seq_id in seq_ids[:num_seqs]]
        if any(idx is None for idx in selected):
            return False
        # The host-side input preparer already checked the worst-case next
        # verify + draft span against max_model_len. Do not inspect the CUDA
        # position tensor here: doing so serializes target graph replay.
        if not get_batch_context().mtp_draft_safe:
            return False
        indices = torch.tensor(
            selected, dtype=torch.long, device=self._prev_drafts.device
        )
        self._selected_prev_drafts = self._prev_drafts.index_select(0, indices)
        return True

    def init_graph_runners(self, target_model, graph_pool, cache_ctx):
        # The legacy one-step path has no predictor KV and can use the original
        # one-token graph. GLM's recurrent top-1 path captures the N-1 decode
        # chain, including its dynamic MLA page metadata and DSA IndexShare.
        if self.config.num_speculative_tokens == 1:
            self.mtp_graph_runner = MTPGraphRunner(self.config, self.config.hf_config)
            self.mtp_graph_runner.capture(self.mtp_model, graph_pool)
        elif (
            self._share_mtp_indexer
            and get_dist_context().attn_tp_world_size == 1
            and self.config.enable_mtp_chain_graph
        ):
            self.cached_mtp_graph_runner = CachedMTPChainGraphRunner(
                self.config, self.config.hf_config
            )
            self.cached_mtp_graph_runner.capture(self.mtp_model, graph_pool)
            torch.cuda.synchronize()
        self.lv_graph_runner = LazyVerifyGraphRunner(
            self.config, self.config.hf_config, cache_ctx
        )
        self.lv_graph_runner.capture(target_model, graph_pool, cache_ctx)
        torch.cuda.synchronize()

    def cleanup(self):
        if self.lv_graph_runner is not None:
            del self.lv_graph_runner
            self.lv_graph_runner = None
        if self.mtp_graph_runner is not None:
            del self.mtp_graph_runner
            self.mtp_graph_runner = None
        if self.cached_mtp_graph_runner is not None:
            del self.cached_mtp_graph_runner
            self.cached_mtp_graph_runner = None

    def prepare_lazy_verify_decode(
        self, input_ids: torch.Tensor, positions: torch.Tensor, num_seqs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand decode to sequence-major ``[base, d1, ..., dN]`` rows."""
        if self._selected_prev_drafts is None:
            raise RuntimeError("lazy verify requested without sequence-aligned drafts")
        sp_rank = get_dist_context().attn_sp_rank
        block_size = self.config.kvcache_block_size
        context = get_batch_context()
        num_drafts = self.config.num_speculative_tokens
        width = num_drafts + 1

        new_input_ids = torch.cat(
            (input_ids[:num_seqs, None], self._selected_prev_drafts[:num_seqs]), dim=1
        ).reshape(-1)
        offsets = torch.arange(width, device=positions.device, dtype=positions.dtype)
        new_positions = (positions[:num_seqs, None] + offsets[None, :]).reshape(-1)

        old_context_lens = context.context_lens[sp_rank, :num_seqs]
        cache_positions = old_context_lens[:, None] - 1 + offsets[None, :]
        block_indices = (cache_positions // block_size).long()
        max_blocks = context.block_tables.shape[2]
        # CUDA decode was admitted by the host-side mtp_draft_safe check and
        # the scheduler-reserved page table. Retain the explicit assertion for
        # CPU/unit-test callers without synchronizing the production stream.
        if not block_indices.is_cuda and bool((block_indices >= max_blocks).any()):
            raise RuntimeError("MTP KV reservation is smaller than the verify width")
        page_ids = context.block_tables[sp_rank, :num_seqs].gather(1, block_indices)
        slot_mapping = page_ids * block_size + (cache_positions % block_size)

        context.context_lens[sp_rank, :num_seqs] += num_drafts
        context.slot_mapping = slot_mapping.to(torch.int32).reshape(-1)
        context.num_tokens_per_seq = width

        if getattr(self.config.hf_config, "kv_lora_rank", 0) > 0:
            import flash_mla

            tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()
            get_hca_context().tile_scheduler_metadata = tile_scheduler_metadata

        return new_input_ids, new_positions

    def lazy_verify_sample(
        self,
        logits: torch.Tensor,
        aux,
        num_seqs: int,
        num_accepted: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Verify all recurrent drafts and sample exactly from the target policy."""
        width = self.verify_width
        target_logits = logits[: num_seqs * width].reshape(
            num_seqs, width, logits.shape[-1]
        )
        drafts = self._selected_prev_drafts[:num_seqs]
        want_logprobs = bool(getattr(aux, "any_return_completion_logprobs", False))
        tp_rank = get_dist_context().attn_tp_rank

        if tp_rank == 0:
            all_greedy = all(
                float(temperature) < 1e-5 for temperature in aux.temperatures[:num_seqs]
            )
            if all_greedy and not want_logprobs:
                input_ids, accepted = linear_greedy_verify(target_logits, drafts)
                next_logprobs = torch.empty(
                    num_seqs, dtype=torch.float32, device=logits.device
                )
                draft_logprobs = torch.empty(
                    num_seqs,
                    self.config.num_speculative_tokens,
                    dtype=torch.float32,
                    device=logits.device,
                )
            else:
                temperatures = prepare_sample_from_aux(aux)
                input_ids, accepted, next_logprobs, draft_logprobs = (
                    linear_rejection_sample(target_logits, drafts, temperatures)
                )
            num_accepted.copy_(accepted)
        else:
            input_ids = torch.zeros(num_seqs, dtype=torch.int64, device=logits.device)
            next_logprobs = torch.zeros(
                num_seqs, dtype=torch.float32, device=logits.device
            )
            draft_logprobs = torch.zeros(
                num_seqs,
                self.config.num_speculative_tokens,
                dtype=torch.float32,
                device=logits.device,
            )

        group = get_dist_context().attn_tp_group
        dist.all_reduce(input_ids, group=group)
        dist.all_reduce(num_accepted, group=group)
        if want_logprobs:
            dist.all_reduce(next_logprobs, group=group)
            dist.all_reduce(draft_logprobs, group=group)

        self._mtp_verified_tokens = drafts.T
        self._mtp_verified_logprobs = draft_logprobs.T if want_logprobs else None
        self._mtp_num_accepted = num_accepted

        # The target forward populated all N draft cache positions. Only the
        # accepted prefix is logically visible to the next decode step.
        context = get_batch_context()
        sp_rank = get_dist_context().attn_sp_rank
        rollback = self.config.num_speculative_tokens - num_accepted
        context.context_lens[sp_rank, :num_seqs] -= rollback.to(torch.int32)

        # Stateful GDN is deliberately limited to the legacy N=1 path.
        rejected_mask = num_accepted == 0
        cache_ctx = get_cache_context()
        # Check the host-side feature gate first. GLM's MLA path has no GDN;
        # evaluating ``rejected_mask.any()`` first forced an otherwise useless
        # device-to-host synchronization after every verification round.
        if cache_ctx.gdn_conv_states is not None and rejected_mask.any():
            active_slots = context.gdn_state_slots[:num_seqs]
            real_slot_mask = active_slots < cache_ctx.gdn_max_active_slots
            rollback_mask = rejected_mask & real_slot_mask
            if rollback_mask.any():
                backup_offset = cache_ctx.gdn_max_active_slots
                rejected_active = active_slots[rollback_mask]
                rejected_backup = rejected_active + backup_offset
                cache_ctx.gdn_conv_states[:, rejected_active] = (
                    cache_ctx.gdn_conv_states[:, rejected_backup]
                )
                cache_ctx.gdn_recurrent_states[:, rejected_active] = (
                    cache_ctx.gdn_recurrent_states[:, rejected_backup]
                )

        return input_ids, next_logprobs if want_logprobs else None

    def generate_and_store(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        aux,
        num_seqs: int,
        has_lazy_verify: bool,
        num_accepted: torch.Tensor | None,
    ) -> None:
        """Run one model-native predictor recurrently N times for next step."""
        decode_context = get_batch_context()
        saved_token_ids = get_batch_out_context().token_ids
        saved_tile_scheduler_metadata = get_hca_context().tile_scheduler_metadata
        saved_sparse_scheduler_metadata = (
            get_mla_context().sparse_tile_scheduler_metadata
        )

        # Attention-DP shards without real requests receive a synthetic decode
        # batch whose page table is intentionally empty. They still must enter
        # all predictor MoE collectives, but cannot use persistent predictor KV.
        # Run the same number of recurrent forwards on an uncached one-token
        # batch so active and dummy EP ranks stay in lockstep.
        if decode_context.is_dummy:
            self.reset_lazy_verify_state()
            self._run_uncached_collective_padding(
                input_ids,
                positions,
                self.last_hidden,
                num_seqs,
            )
            self._restore_batch_context(decode_context)
            get_hca_context().tile_scheduler_metadata = saved_tile_scheduler_metadata
            get_mla_context().sparse_tile_scheduler_metadata = (
                saved_sparse_scheduler_metadata
            )
            get_batch_out_context().token_ids = saved_token_ids
            return

        if has_lazy_verify:
            verify_hidden = self.last_hidden.reshape(
                num_seqs, self.verify_width, self.last_hidden.shape[-1]
            )
            rows = torch.arange(num_seqs, device=verify_hidden.device)
            mtp_hidden = verify_hidden[rows, num_accepted]
            verify_positions = positions.reshape(num_seqs, self.verify_width)
            base_positions = verify_positions[:, 0]
            mtp_positions = base_positions + num_accepted
        else:
            verify_hidden = None
            verify_positions = None
            mtp_hidden = self.last_hidden
            mtp_positions = positions

        # The recurrent predictor would otherwise produce draft tokens whose
        # positions cannot all be consumed by the next fixed-width verify. A
        # single near-limit sequence makes this batch use ordinary decode on
        # the next step; keep this step's verified output state intact.
        if not decode_context.mtp_draft_safe:
            self._prev_seq_ids = None
            self._prev_drafts = None
            self._selected_prev_drafts = None
            return

        if self.config.num_speculative_tokens > 1:
            sp_rank = get_dist_context().attn_sp_rank
            target_lens = decode_context.context_lens[sp_rank, :num_seqs]
            if has_lazy_verify:
                drafts = self._refresh_and_generate_cached_mtp_drafts(
                    input_ids,
                    verify_positions,
                    verify_hidden,
                    target_lens,
                    num_accepted,
                    decode_context,
                    num_seqs,
                )
            else:
                drafts = self._generate_cached_mtp_drafts(
                    input_ids,
                    mtp_positions + 1,
                    mtp_hidden,
                    target_lens - 1,
                    decode_context,
                    num_seqs,
                )
        else:
            drafts = self._generate_mtp_drafts(
                input_ids, mtp_positions, mtp_hidden, num_seqs
            )
        self._prev_seq_ids = tuple(int(seq_id) for seq_id in aux.seq_ids[:num_seqs])
        self._prev_drafts = torch.stack(drafts, dim=1)
        self._selected_prev_drafts = None

        set_batch_context(
            is_prefill=decode_context.is_prefill,
            max_bs=decode_context.max_bs,
            cu_seqlens_q=decode_context.cu_seqlens_q,
            cu_seqlens_k=decode_context.cu_seqlens_k,
            max_seqlen_q=decode_context.max_seqlen_q,
            max_seqlen_k=decode_context.max_seqlen_k,
            slot_mapping=decode_context.slot_mapping,
            context_lens=decode_context.context_lens,
            block_tables=decode_context.block_tables,
            is_dummy=decode_context.is_dummy,
            gdn_conv_states=decode_context.gdn_conv_states,
            gdn_recurrent_states=decode_context.gdn_recurrent_states,
            gdn_state_slots=decode_context.gdn_state_slots,
            dsv4_state_slots=decode_context.dsv4_state_slots,
            dsv4_compressed_block_tables=decode_context.dsv4_compressed_block_tables,
            hisparse_slots=decode_context.hisparse_slots,
            hisparse_slot_mapping=decode_context.hisparse_slot_mapping,
            hisparse_num_real_reqs=decode_context.hisparse_num_real_reqs,
            hisparse_phase_id=decode_context.hisparse_phase_id,
            num_tokens_per_seq=decode_context.num_tokens_per_seq,
            sampling_token_indices=decode_context.sampling_token_indices,
            sampling_seq_indices=decode_context.sampling_seq_indices,
            paged_attention_strategy=decode_context.paged_attention_strategy,
            graph_attention_strategy=decode_context.graph_attention_strategy,
            decode_page_plan_key=decode_context.decode_page_plan_key,
            mtp_draft_safe=decode_context.mtp_draft_safe,
        )
        get_hca_context().tile_scheduler_metadata = saved_tile_scheduler_metadata
        get_mla_context().sparse_tile_scheduler_metadata = (
            saved_sparse_scheduler_metadata
        )
        get_batch_out_context().token_ids = saved_token_ids

    def _select_prefill_seed_batch(
        self,
        target_input_ids: torch.Tensor,
        target_positions: torch.Tensor,
        sampled_ids: torch.Tensor,
        aux,
        context,
        num_seqs: int,
    ):
        """Compact completed rows from a mixed PP prefill microbatch."""
        if self.last_hidden is None or sampled_ids.numel() < num_seqs:
            return None

        active_cu_q = context.cu_seqlens_q
        active_cu_k = context.cu_seqlens_k
        if (
            active_cu_q is None
            or active_cu_k is None
            or context.block_tables is None
            or context.slot_mapping is None
        ):
            return None

        row_indices = context.sampling_seq_indices
        if row_indices is None:
            return (
                target_input_ids,
                target_positions,
                sampled_ids,
                self.last_hidden,
                context,
                tuple(int(seq_id) for seq_id in aux.seq_ids[:num_seqs]),
                num_seqs,
            )
        if row_indices.numel() == 0:
            return None

        active_cu_q = active_cu_q[: num_seqs + 1]
        active_cu_k = active_cu_k[: num_seqs + 1]
        row_indices = row_indices.to(device=active_cu_q.device, dtype=torch.long)
        token_indices, compact_cu_q = _select_ragged_rows(active_cu_q, row_indices)
        if token_indices.numel() == 0:
            return None

        key_lengths = active_cu_k[1:] - active_cu_k[:-1]
        compact_cu_k = torch.cat(
            (
                active_cu_k.new_zeros(1),
                key_lengths.index_select(0, row_indices).cumsum(0),
            )
        )
        selected_context_lens = context.context_lens
        if selected_context_lens is not None and selected_context_lens.ndim >= 2:
            selected_context_lens = selected_context_lens.index_select(1, row_indices)
        selected_hisparse_slots = context.hisparse_slots
        if selected_hisparse_slots is not None:
            selected_hisparse_slots = selected_hisparse_slots.index_select(
                0, row_indices
            )
        seed_context = replace(
            context,
            cu_seqlens_q=compact_cu_q,
            cu_seqlens_k=compact_cu_k,
            slot_mapping=context.slot_mapping.index_select(0, token_indices),
            context_lens=selected_context_lens,
            block_tables=context.block_tables.index_select(1, row_indices),
            hisparse_slots=selected_hisparse_slots,
            sampling_token_indices=None,
            sampling_seq_indices=None,
        )
        host_rows = [int(row) for row in row_indices.tolist()]
        return (
            target_input_ids.index_select(0, token_indices),
            target_positions.index_select(0, token_indices),
            sampled_ids.index_select(0, row_indices),
            self.last_hidden.index_select(0, token_indices),
            seed_context,
            tuple(int(aux.seq_ids[row]) for row in host_rows),
            len(host_rows),
        )

    def generate_prefill_and_store(
        self,
        target_input_ids: torch.Tensor,
        target_positions: torch.Tensor,
        sampled_ids: torch.Tensor,
        aux,
        num_seqs: int,
    ) -> None:
        """Seed persistent GLM predictor KV from the target prefill.

        For target hidden state h[t], NextN consumes token x[t+1].
        Therefore each fresh prefill segment is shifted left and terminated by
        the target-sampled token. Predictor cache slots deliberately reuse the
        target segment slot mapping: logical predictor slot zero represents
        absolute token position one.
        """
        if self.config.num_speculative_tokens == 1:
            self.reset_lazy_verify_state()
            return

        target_context = get_batch_context()
        saved_token_ids = get_batch_out_context().token_ids
        saved_tile_scheduler_metadata = get_hca_context().tile_scheduler_metadata
        saved_sparse_scheduler_metadata = (
            get_mla_context().sparse_tile_scheduler_metadata
        )

        seed_batch = self._select_prefill_seed_batch(
            target_input_ids,
            target_positions,
            sampled_ids,
            aux,
            target_context,
            num_seqs,
        )
        if seed_batch is not None:
            (
                target_input_ids,
                target_positions,
                sampled_ids,
                seed_hidden,
                context,
                seed_seq_ids,
                num_seqs,
            ) = seed_batch
            # A full prefix-cache hit may have a zero-length fresh segment.
            # ``cu_q[1:] - 1`` would then produce -1 and crash the CUDA
            # index-select used to seed recurrent MTP. Skip drafts for this
            # round; ordinary decode will seed them on the next target token.
            if _nonempty_ragged_bounds(context.cu_seqlens_q, num_seqs) is None:
                seed_batch = None
        if seed_batch is None:
            self.reset_lazy_verify_state()
            self._run_uncached_collective_padding(
                target_input_ids,
                target_positions,
                self.last_hidden,
                num_seqs,
            )
            self._restore_batch_context(target_context)
            get_hca_context().tile_scheduler_metadata = saved_tile_scheduler_metadata
            get_mla_context().sparse_tile_scheduler_metadata = (
                saved_sparse_scheduler_metadata
            )
            get_batch_out_context().token_ids = saved_token_ids
            return

        cu_q = context.cu_seqlens_q
        shifted_parts = []
        for seq_idx in range(num_seqs):
            start = int(cu_q[seq_idx].item())
            end = int(cu_q[seq_idx + 1].item())
            shifted_parts.append(
                torch.cat(
                    (
                        target_input_ids[start + 1 : end],
                        sampled_ids[seq_idx : seq_idx + 1],
                    )
                )
            )
        shifted_ids = torch.cat(shifted_parts)
        shifted_positions = target_positions + 1

        # Use the target prefill geometry while writing the extra predictor
        # cache layer. This supports ordinary and prefix-cached fresh segments.
        active_cu_q = context.cu_seqlens_q[: num_seqs + 1]
        active_cu_k = context.cu_seqlens_k[: num_seqs + 1]
        set_batch_context(
            is_prefill=True,
            max_bs=context.max_bs,
            # InputPreparer owns max-batch-sized buffers. DSA derives its
            # sequence count from the cu-seqlens shape, so exposing the padded
            # suffix makes it index nonexistent MTP input rows at batch > 1.
            cu_seqlens_q=active_cu_q,
            cu_seqlens_k=active_cu_k,
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            slot_mapping=context.slot_mapping,
            block_tables=context.block_tables,
            is_dummy=context.is_dummy,
            num_tokens_per_seq=1,
            paged_attention_strategy=context.paged_attention_strategy,
        )
        set_expert_context(use_low_latency_ep=True)
        indexer_state = self._new_mtp_indexer_state()
        mtp_hidden = self._forward_cached_mtp(
            shifted_ids,
            shifted_positions,
            seed_hidden,
            0,
            indexer_state,
        )

        # Batch contexts keep max-batch-sized cu-seqlens buffers. Consuming
        # the padded suffix produces -1 rows and a CUDA ScatterGather OOB as
        # soon as a rank prefills more than one request after cold start.
        last_rows = _active_ragged_last_rows(cu_q, num_seqs)
        last_hidden = mtp_hidden.index_select(0, last_rows)
        last_positions = shifted_positions.index_select(0, last_rows)
        indexer_state = self._select_mtp_indexer_seed(
            indexer_state,
            last_rows,
            context,
            num_seqs,
            packed_k_starts=active_cu_k[:-1],
        )
        first_logits = self.mtp_model.compute_logits(last_hidden, spec_step_idx=0)
        first_draft = self._greedy_draft(first_logits, sampled_ids)

        total_lens = (active_cu_k[1:] - active_cu_k[:-1]).to(torch.int32)
        drafts = [first_draft]
        drafts.extend(
            self._continue_cached_mtp_drafts(
                first_draft,
                last_positions + 1,
                last_hidden,
                total_lens,
                context,
                num_seqs,
                self.config.num_speculative_tokens - 1,
                start_step=1,
                indexer_state=indexer_state,
            )
        )

        self._prev_seq_ids = seed_seq_ids
        self._prev_drafts = torch.stack(drafts, dim=1)
        self._selected_prev_drafts = None
        self._restore_batch_context(target_context)
        get_hca_context().tile_scheduler_metadata = saved_tile_scheduler_metadata
        get_mla_context().sparse_tile_scheduler_metadata = (
            saved_sparse_scheduler_metadata
        )
        get_batch_out_context().token_ids = saved_token_ids

    def build_output_logprobs(
        self, next_logprobs: list[list[float]]
    ) -> list[list[float]]:
        """Prepend accepted-draft target logprobs to each next-token logprob."""
        if self._mtp_verified_logprobs is None or self._mtp_num_accepted is None:
            return next_logprobs
        result: list[list[float]] = []
        for seq_idx, base in enumerate(next_logprobs):
            accepted = int(self._mtp_num_accepted[seq_idx].item())
            prefix = self._mtp_verified_logprobs[:accepted, seq_idx].tolist()
            result.append([float(value) for value in prefix] + base)
        return result

    def build_output_tokens(self, rank: int) -> list[list[int]]:
        """Prepend accepted draft tokens to the newly sampled target token."""
        token_ids = get_batch_out_context().token_ids
        if self._mtp_verified_tokens is None:
            return torch.cat(token_ids, dim=0).T.tolist()

        base = torch.cat(token_ids, dim=0)
        result = []
        for seq_idx in range(base.shape[1]):
            accepted = int(self._mtp_num_accepted[seq_idx].item())
            prefix = self._mtp_verified_tokens[:accepted, seq_idx].tolist()
            result.append(
                [int(token) for token in prefix]
                + [int(token) for token in base[:, seq_idx].tolist()]
            )
        if rank == 0 and result:
            logger.debug("MTP output tokens[0]=%s", result[0])
        self._mtp_verified_tokens = None
        self._mtp_verified_logprobs = None
        self._mtp_num_accepted = None
        return result

    def _set_mtp_context(self, num_seqs: int):
        cu_seqlens = torch.arange(num_seqs + 1, dtype=torch.int32, device="cuda")
        set_batch_context(
            is_prefill=True,
            max_bs=self.config.max_num_seqs,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=1,
            max_seqlen_k=1,
            slot_mapping=None,
            block_tables=None,
            is_dummy=False,
        )
        set_expert_context(use_low_latency_ep=True)

    def _restore_batch_context(self, context) -> None:
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

    def _set_cached_mtp_context(
        self,
        source_context,
        cache_lens: torch.Tensor,
        num_seqs: int,
        phase_id: int,
    ) -> None:
        """Expose the predictor layer to the shared page table at given lengths."""
        sp_rank = get_dist_context().attn_sp_rank
        block_size = self.config.kvcache_block_size
        block_tables = source_context.block_tables
        if block_tables is None:
            raise RuntimeError("cached GLM MTP requires paged block tables")

        logical_slots = cache_lens[:num_seqs].to(torch.long) - 1
        block_indices = logical_slots // block_size
        page_ids = block_tables[sp_rank, :num_seqs].gather(1, block_indices[:, None])[
            :, 0
        ]
        slot_mapping = (page_ids * block_size + (logical_slots % block_size)).to(
            torch.int32
        )

        if source_context.context_lens is not None:
            context_lens = source_context.context_lens.clone()
        else:
            sp_size = block_tables.shape[0]
            context_lens = torch.zeros(
                sp_size,
                self.config.max_num_seqs,
                dtype=torch.int32,
                device=cache_lens.device,
            )
        context_lens[sp_rank, :num_seqs] = cache_lens[:num_seqs].to(torch.int32)

        set_batch_context(
            is_prefill=False,
            max_bs=source_context.max_bs,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            is_dummy=source_context.is_dummy,
            num_tokens_per_seq=1,
            paged_attention_strategy=PagedAttentionStrategy.FLASH_MLA,
            hisparse_slots=source_context.hisparse_slots,
            hisparse_num_real_reqs=source_context.hisparse_num_real_reqs,
            hisparse_phase_id=phase_id,
        )
        if torch.cuda.get_device_capability()[0] < 10:
            import flash_mla

            tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()
            get_hca_context().tile_scheduler_metadata = tile_scheduler_metadata
            sparse_scheduler_metadata, _ = flash_mla.get_mla_metadata()
            get_mla_context().sparse_tile_scheduler_metadata = sparse_scheduler_metadata
        set_expert_context(use_low_latency_ep=True)

    def _greedy_draft(
        self, logits: torch.Tensor, template_ids: torch.Tensor
    ) -> torch.Tensor:
        if get_dist_context().attn_tp_rank == 0:
            draft_ids = logits.argmax(dim=-1)
        else:
            draft_ids = template_ids.new_zeros(logits.shape[0])
        dist.all_reduce(draft_ids, group=get_dist_context().attn_tp_group)
        return draft_ids

    def _forward_cached_mtp(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        spec_step_idx: int,
        indexer_state: _IndexerTopKState | None,
        *,
        reuse_indexer_topk: bool = False,
    ) -> torch.Tensor:
        """Forward a cached predictor, adding GLM-only DSA state when enabled."""
        if indexer_state is None:
            return self.mtp_model(
                input_ids,
                positions,
                hidden_states,
                spec_step_idx=spec_step_idx,
            )
        return self.mtp_model(
            input_ids,
            positions,
            hidden_states,
            spec_step_idx=spec_step_idx,
            indexer_state=indexer_state,
            reuse_indexer_topk=reuse_indexer_topk,
        )

    def _continue_cached_mtp_drafts(
        self,
        current_ids: torch.Tensor,
        current_positions: torch.Tensor,
        current_hidden: torch.Tensor,
        initial_cache_lens: torch.Tensor,
        source_context,
        num_seqs: int,
        num_steps: int,
        *,
        start_step: int,
        indexer_state: _IndexerTopKState | None = None,
    ) -> list[torch.Tensor]:
        if (
            self.cached_mtp_graph_runner is not None
            and start_step == 1
            and num_steps == self.config.num_speculative_tokens - 1
        ):
            graph_drafts = self.cached_mtp_graph_runner.run(
                current_ids,
                current_positions,
                current_hidden,
                initial_cache_lens,
                source_context,
                indexer_state,
                num_seqs,
            )
            if graph_drafts is not None:
                return graph_drafts

        drafts = []
        cache_lens = initial_cache_lens.to(torch.int32).clone()
        if indexer_state is None:
            indexer_state = self._new_mtp_indexer_state()
        for offset in range(num_steps):
            cache_lens += 1
            step_idx = start_step + offset
            self._set_cached_mtp_context(
                source_context, cache_lens, num_seqs, step_idx + 1
            )
            mtp_hidden = self._forward_cached_mtp(
                current_ids,
                current_positions,
                current_hidden,
                step_idx,
                indexer_state,
                reuse_indexer_topk=(
                    indexer_state is not None
                    and indexer_state.logical_indices is not None
                ),
            )
            mtp_logits = self.mtp_model.compute_logits(
                mtp_hidden, spec_step_idx=step_idx
            )
            draft_ids = self._greedy_draft(mtp_logits, current_ids)
            drafts.append(draft_ids)
            current_hidden = mtp_hidden
            current_ids = draft_ids
            current_positions += 1
        return drafts

    def _generate_cached_mtp_drafts(
        self,
        sampled_ids: torch.Tensor,
        seed_positions: torch.Tensor,
        hidden_states: torch.Tensor,
        initial_cache_lens: torch.Tensor,
        source_context,
        num_seqs: int,
    ) -> list[torch.Tensor]:
        return self._continue_cached_mtp_drafts(
            sampled_ids,
            seed_positions,
            hidden_states,
            initial_cache_lens,
            source_context,
            num_seqs,
            self.config.num_speculative_tokens,
            start_step=0,
        )

    def _refresh_and_generate_cached_mtp_drafts(
        self,
        sampled_ids: torch.Tensor,
        verify_positions: torch.Tensor,
        verify_hidden: torch.Tensor,
        target_lens: torch.Tensor,
        num_accepted: torch.Tensor,
        source_context,
        num_seqs: int,
    ) -> list[torch.Tensor]:
        """Refresh accepted predictor KV with target hidden states, then draft.

        Recurrent drafting has to approximate unavailable future target hidden
        states with predictor hidden states. After verification, replay the
        accepted line plus the newly sampled token through NextN using the
        exact target hidden rows. This draft-extend stage overwrites speculative
        predictor KV before the next recurrence and is essential for GLM
        acceptance quality.
        """
        if self._selected_prev_drafts is None:
            raise RuntimeError("MTP cache refresh requires aligned verified drafts")

        sp_rank = get_dist_context().attn_sp_rank
        block_size = self.config.kvcache_block_size
        block_tables = source_context.block_tables
        if block_tables is None:
            raise RuntimeError("MTP cache refresh requires paged block tables")

        width = self.verify_width
        refresh_lens = num_accepted.to(torch.int32) + 1
        prefix_lens = target_lens.to(torch.int32) - refresh_lens

        # Match the target verify geometry: replay all K rows in one causal
        # multi-token decode and select only the row at the accepted length.
        # Rows after the selected token are disposable; fixed width keeps MLA
        # and DSA metadata identical across the batch.
        refresh_candidates = torch.cat(
            (
                self._selected_prev_drafts[:num_seqs],
                sampled_ids[:num_seqs, None],
            ),
            dim=1,
        )
        refresh_offsets = torch.arange(width, device=sampled_ids.device)
        refresh_ids = torch.where(
            refresh_offsets[None, :] < num_accepted[:num_seqs, None],
            refresh_candidates,
            sampled_ids[:num_seqs, None],
        )

        refresh_positions = verify_positions + 1
        logical_slots = (
            prefix_lens.to(torch.long)[:, None]
            + torch.arange(
                width,
                dtype=torch.long,
                device=sampled_ids.device,
            )[None, :]
        )
        block_indices = logical_slots // block_size
        page_ids = block_tables[sp_rank, :num_seqs].gather(1, block_indices)
        slot_mapping = (page_ids * block_size + (logical_slots % block_size)).to(
            torch.int32
        )

        refresh_context_lens = source_context.context_lens.clone()
        full_refresh_lens = prefix_lens + width
        refresh_context_lens[sp_rank, :num_seqs] = full_refresh_lens

        set_batch_context(
            is_prefill=False,
            max_bs=source_context.max_bs,
            slot_mapping=slot_mapping.reshape(-1),
            context_lens=refresh_context_lens,
            block_tables=block_tables,
            is_dummy=source_context.is_dummy,
            num_tokens_per_seq=width,
            paged_attention_strategy=PagedAttentionStrategy.FLASH_MLA,
            hisparse_slots=source_context.hisparse_slots,
            hisparse_num_real_reqs=source_context.hisparse_num_real_reqs,
            hisparse_phase_id=self.verify_width,
        )
        if torch.cuda.get_device_capability()[0] < 10:
            import flash_mla

            tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()
            get_hca_context().tile_scheduler_metadata = tile_scheduler_metadata
            sparse_scheduler_metadata, _ = flash_mla.get_mla_metadata()
            get_mla_context().sparse_tile_scheduler_metadata = sparse_scheduler_metadata
        set_expert_context(use_low_latency_ep=True)
        indexer_state = self._new_mtp_indexer_state()
        refreshed_hidden = self._forward_cached_mtp(
            refresh_ids.reshape(-1),
            refresh_positions.reshape(-1),
            verify_hidden.reshape(-1, verify_hidden.shape[-1]),
            0,
            indexer_state,
        )

        last_rows = (
            torch.arange(num_seqs, device=sampled_ids.device) * width + num_accepted
        )
        last_hidden = refreshed_hidden.index_select(0, last_rows)
        indexer_state = self._select_mtp_indexer_seed(
            indexer_state,
            last_rows,
            source_context,
            num_seqs,
        )
        rows = torch.arange(num_seqs, device=sampled_ids.device)
        last_positions = refresh_positions[rows, num_accepted]
        first_logits = self.mtp_model.compute_logits(last_hidden, spec_step_idx=0)
        first_draft = self._greedy_draft(first_logits, sampled_ids)

        drafts = [first_draft]
        drafts.extend(
            self._continue_cached_mtp_drafts(
                first_draft,
                last_positions + 1,
                last_hidden,
                target_lens,
                source_context,
                num_seqs,
                self.config.num_speculative_tokens - 1,
                start_step=1,
                indexer_state=indexer_state,
            )
        )
        return drafts

    def _run_uncached_collective_padding(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None,
        num_seqs: int,
    ) -> None:
        """Keep EP collective counts aligned when a prefill cannot seed KV."""
        take = max(1, num_seqs)
        device = input_ids.device

        if input_ids.numel() >= take:
            ids = input_ids[:take]
        else:
            ids = torch.zeros(take, dtype=input_ids.dtype, device=device)
            if input_ids.numel():
                ids[: input_ids.numel()].copy_(input_ids)

        if positions.numel() >= take:
            pos = positions[:take] + 1
        else:
            pos = torch.ones(take, dtype=positions.dtype, device=device)
            if positions.numel():
                pos[: positions.numel()].copy_(positions + 1)

        hidden_dtype = (
            hidden_states.dtype
            if hidden_states is not None
            else torch.get_default_dtype()
        )
        if hidden_states is not None and hidden_states.size(0) >= take:
            hidden = hidden_states[:take]
        else:
            hidden = torch.zeros(
                take,
                self.config.hf_config.hidden_size,
                dtype=hidden_dtype,
                device=device,
            )
            if hidden_states is not None and hidden_states.size(0):
                count = min(hidden_states.size(0), take)
                hidden[:count].copy_(hidden_states[:count])
        for step in range(self.config.num_speculative_tokens):
            self._set_mtp_context(ids.numel())
            hidden = self.mtp_model(ids, pos, hidden, spec_step_idx=step)
            logits = self.mtp_model.compute_logits(hidden, spec_step_idx=step)
            ids = self._greedy_draft(logits, ids)
            pos += 1
            if step == 0:
                chain_graph = getattr(self, "cached_mtp_graph_runner", None)
                if chain_graph is not None and chain_graph.run_padding(
                    ids, pos, hidden, ids.numel()
                ):
                    return

    def _run_mtp_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        spec_step_idx: int,
        batch_size: int,
    ) -> torch.Tensor:
        if self.mtp_graph_runner is not None:
            output = self.mtp_graph_runner.run(
                input_ids,
                positions,
                hidden_states,
                spec_step_idx,
                batch_size,
            )
            if output is not None:
                return output
        self._set_mtp_context(batch_size)
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
        num_seqs: int,
    ) -> list[torch.Tensor]:
        drafts = []
        current_hidden = hidden_states
        current_ids = sampled_ids
        current_pos = positions + 1

        for step in range(self.config.num_speculative_tokens):
            mtp_hidden = self._run_mtp_step(
                current_ids, current_pos, current_hidden, step, num_seqs
            )
            mtp_logits = self.mtp_model.compute_logits(mtp_hidden, spec_step_idx=step)

            if get_dist_context().attn_tp_rank == 0:
                # The model-native draft policy is deliberately one-hot. This
                # makes rejection sampling exact without retaining draft logits.
                draft_ids = mtp_logits.argmax(dim=-1)
            else:
                draft_ids = current_ids.new_zeros(num_seqs)
            dist.all_reduce(draft_ids, group=get_dist_context().attn_tp_group)

            drafts.append(draft_ids)
            current_hidden = mtp_hidden
            current_ids = draft_ids
            current_pos += 1

        return drafts
