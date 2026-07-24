from __future__ import annotations

import torch

from nanodeploy._cpp import (
    BlockContextSlot,
    Sequence,
    SequenceStatus,
    deserialize,
    prepare_decode_cpp,
    serialize,
)


_BUFFER_BYTES = 1 << 20
_SP_RANK = 1
_SP_SIZE = 2
_BLOCK_SIZE = 16
_MAX_NUM_SEQS = 4

_META_FIELDS = (
    "use_sp_a2a",
    "input_ids",
    "positions",
    "slot_mapping",
    "context_lens_flat",
    "global_context_lens_flat",
    "block_tables_flat",
    "max_num_blocks",
    "context_lens_for_attn",
    "q_slice_get",
    "q_slice_fill",
    "q_copy_mask",
    "res_slice_get_to_buffer_output",
    "res_slice_fill_to_buffer_output",
    "res_to_buffer_output_mask",
    "res_slice_get_to_buffer_input",
    "res_slice_fill_to_buffer_input",
    "res_to_buffer_input_mask",
    "q_offsets",
)


def _make_seq(
    *,
    seq_id: int,
    master_sp_idx: int,
    num_dispatched_tokens: list[int],
    sp_block_tables: dict[int, list[int]],
    block_location: list[tuple[int, int]],
) -> Sequence:
    seq = Sequence([1000 + seq_id], 0.7 + seq_id * 0.1, 64 + seq_id, False)
    seq.seq_id = seq_id
    seq.status = SequenceStatus.RUNNING
    seq.last_token = 2000 + seq_id
    seq.num_tokens = 10 + seq_id
    seq.num_prompt_tokens = 7 + seq_id
    seq.num_bootstrap_tokens = seq_id % 2
    seq.num_checkpointed_tokens = 6 + seq_id
    seq.num_cached_tokens = seq_id
    seq.active("decode-engine", _SP_SIZE, 1)

    ctx = seq.block_ctx(BlockContextSlot.ACTIVE)
    ctx.dp_idx = 0
    ctx.master_sp_idx = master_sp_idx
    ctx.num_dispatched_tokens = num_dispatched_tokens
    ctx.block_location.clear()
    for pair in block_location:
        ctx.block_location.append(pair)

    for sp_idx in range(_SP_SIZE):
        table = ctx.sp_block_table[sp_idx]
        table.clear()
        for block_id in sp_block_tables.get(sp_idx, []):
            table.append(block_id)

    return seq


def _build_regression_case() -> list[Sequence]:
    # This layout reproduces the broken pre-fix decode optimize behavior:
    # the first master-0 sequence has ctx_len=0 on sp_rank=1, so filtering the
    # sequence set would left-shift the master-local seq_id and corrupt
    # res_slice_fill_to_buffer_input/context_lens_flat.
    return [
        _make_seq(
            seq_id=10,
            master_sp_idx=0,
            num_dispatched_tokens=[1, 0],
            sp_block_tables={0: [100]},
            block_location=[(0, 100), (1, 999)],
        ),
        _make_seq(
            seq_id=11,
            master_sp_idx=0,
            num_dispatched_tokens=[1, 2],
            sp_block_tables={0: [110], 1: [210]},
            block_location=[(0, 110), (1, 210)],
        ),
        _make_seq(
            seq_id=12,
            master_sp_idx=1,
            num_dispatched_tokens=[1, 4],
            sp_block_tables={0: [120], 1: [220]},
            block_location=[(0, 120), (1, 220)],
        ),
        _make_seq(
            seq_id=13,
            master_sp_idx=1,
            num_dispatched_tokens=[0, 2],
            sp_block_tables={1: [230]},
            block_location=[(1, 230)],
        ),
    ]


def _serialize_roundtrip(
    seqs: list[Sequence], *, is_prefill: bool, sp_rank: int, sp_size: int
) -> list[Sequence]:
    buffer = torch.empty(_BUFFER_BYTES, dtype=torch.int8)
    data_len = serialize(
        buffer.data_ptr(),
        buffer.numel(),
        seqs,
        is_prefill,
        sp_rank,
        sp_size,
    )
    return deserialize(buffer.data_ptr(), data_len)


def _snapshot_decode_meta(seqs: list[Sequence]) -> dict[str, object]:
    meta = prepare_decode_cpp(seqs, _SP_RANK, _SP_SIZE, _BLOCK_SIZE, _MAX_NUM_SEQS)
    return {name: getattr(meta, name) for name in _META_FIELDS}


def test_decode_optimize_roundtrip_preserves_sequence_skeleton_and_metadata():
    seqs = _build_regression_case()

    baseline = _snapshot_decode_meta(seqs)
    roundtrip = _serialize_roundtrip(
        seqs,
        is_prefill=False,
        sp_rank=_SP_RANK,
        sp_size=_SP_SIZE,
    )

    assert [seq.seq_id for seq in roundtrip] == [seq.seq_id for seq in seqs]
    assert len(roundtrip) == len(seqs)
    assert _snapshot_decode_meta(roundtrip) == baseline


def test_decode_optimize_roundtrip_trims_only_target_rank_heavy_fields():
    seqs = _build_regression_case()
    roundtrip = _serialize_roundtrip(
        seqs,
        is_prefill=False,
        sp_rank=_SP_RANK,
        sp_size=_SP_SIZE,
    )

    for original, restored in zip(seqs, roundtrip, strict=True):
        original_ctx = original.block_ctx(BlockContextSlot.ACTIVE)
        restored_ctx = restored.block_ctx(BlockContextSlot.ACTIVE)

        assert restored.seq_id == original.seq_id
        assert restored.status == original.status
        assert restored.temperature == original.temperature
        assert restored.max_tokens == original.max_tokens
        assert restored.ignore_eos == original.ignore_eos
        assert restored.last_token == original.last_token
        assert restored.num_tokens == original.num_tokens
        assert restored.num_prompt_tokens == original.num_prompt_tokens
        assert (
            restored.num_bootstrap_tokens == original.num_bootstrap_tokens
        )
        assert restored.num_checkpointed_tokens == original.num_checkpointed_tokens
        assert restored.num_cached_tokens == original.num_cached_tokens

        assert restored_ctx.engine_id == original_ctx.engine_id
        assert restored_ctx.dp_idx == original_ctx.dp_idx
        assert restored_ctx.master_sp_idx == original_ctx.master_sp_idx
        assert restored_ctx.attention_sp == original_ctx.attention_sp
        assert restored_ctx.attention_dp == original_ctx.attention_dp
        assert restored_ctx.num_dispatched_tokens == original_ctx.num_dispatched_tokens

        assert list(restored_ctx.sp_block_table[_SP_RANK]) == list(
            original_ctx.sp_block_table[_SP_RANK]
        )
        assert list(restored_ctx.sp_block_table[0]) == []
        assert list(restored_ctx.block_location) == [
            pair for pair in list(original_ctx.block_location) if pair[0] == _SP_RANK
        ]
        assert restored.token_ids == []


def test_decode_non_optimized_roundtrip_keeps_full_block_context():
    seqs = _build_regression_case()
    roundtrip = _serialize_roundtrip(
        seqs,
        is_prefill=False,
        sp_rank=-1,
        sp_size=-1,
    )

    for original, restored in zip(seqs, roundtrip, strict=True):
        original_ctx = original.block_ctx(BlockContextSlot.ACTIVE)
        restored_ctx = restored.block_ctx(BlockContextSlot.ACTIVE)

        assert list(restored_ctx.block_location) == list(original_ctx.block_location)
        for sp_idx in range(_SP_SIZE):
            assert list(restored_ctx.sp_block_table[sp_idx]) == list(
                original_ctx.sp_block_table[sp_idx]
            )
