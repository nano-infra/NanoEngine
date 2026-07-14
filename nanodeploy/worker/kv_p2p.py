from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Iterable, Sequence

import torch
import torch.distributed as dist


@dataclasses.dataclass(frozen=True, slots=True)
class KVCacheP2PMove:
    """A physical KV token-range copy within one attention-DP group.

    SP ranks are local to ``attn_sp_group``.  The move describes physical
    cache locations only; it deliberately carries no scheduler metadata and
    never implies that the destination range is active.
    """

    dp_idx: int
    src_sp_rank: int
    dst_sp_rank: int
    src_block_id: int
    src_token_offset: int
    dst_block_id: int
    dst_token_offset: int
    num_tokens: int


@dataclasses.dataclass(frozen=True, slots=True)
class KVCacheP2PResult:
    dp_idx: int
    sp_rank: int
    role: str
    num_moves: int
    num_chunks: int
    sent_bytes: int
    received_bytes: int


class KVCacheP2PTransport:
    """Chunked source-gather / NCCL-P2P / destination-scatter transport.

    One transport owns one fixed scratch tensor.  A plan is restricted to one
    source rank, matching the first KV-consolidation MVP.  All workers receive
    the same plan, while workers outside the selected DP or source/destination
    ranks perform no P2P operation.
    """

    def __init__(
        self,
        kv_cache: torch.Tensor,
        group: dist.ProcessGroup,
        chunk_tokens: int,
        scratch: torch.Tensor | None = None,
    ) -> None:
        if kv_cache.ndim != 6:
            raise ValueError(
                "KV cache must have shape " "[kv, layer, block, token, head, dim]"
            )
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be > 0")

        self.kv_cache = kv_cache
        self.group = group
        self.chunk_tokens = chunk_tokens
        self.block_size = int(kv_cache.size(3))
        self.token_bytes = (
            int(kv_cache.size(0))
            * int(kv_cache.size(1))
            * int(kv_cache.size(4))
            * int(kv_cache.size(5))
            * kv_cache.element_size()
        )

        expected_shape = self.scratch_shape(kv_cache, chunk_tokens)
        if scratch is None:
            scratch = torch.empty(
                expected_shape, dtype=kv_cache.dtype, device=kv_cache.device
            )
        if tuple(scratch.shape) != expected_shape:
            raise ValueError(
                f"scratch shape must be {expected_shape}, got {tuple(scratch.shape)}"
            )
        if scratch.dtype != kv_cache.dtype or scratch.device != kv_cache.device:
            raise ValueError("scratch dtype/device must match KV cache")
        if not scratch.is_contiguous():
            raise ValueError("scratch must be contiguous")
        self.scratch = scratch

    @staticmethod
    def scratch_shape(
        kv_cache: torch.Tensor, chunk_tokens: int
    ) -> tuple[int, int, int, int, int]:
        # Token-major layout keeps scratch[:num_tokens] contiguous for NCCL.
        return (
            chunk_tokens,
            int(kv_cache.size(0)),
            int(kv_cache.size(1)),
            int(kv_cache.size(4)),
            int(kv_cache.size(5)),
        )

    @torch.inference_mode()
    def execute(
        self, moves: Sequence[KVCacheP2PMove], *, current_dp_idx: int
    ) -> KVCacheP2PResult:
        normalized = self._validate_and_normalize(moves)
        sp_rank = dist.get_rank(self.group)

        if not normalized or normalized[0].dp_idx != current_dp_idx:
            return KVCacheP2PResult(
                dp_idx=current_dp_idx,
                sp_rank=sp_rank,
                role="idle",
                num_moves=0,
                num_chunks=0,
                sent_bytes=0,
                received_bytes=0,
            )

        pair_moves: dict[tuple[int, int], list[KVCacheP2PMove]] = defaultdict(list)
        for move in self._split_large_moves(normalized):
            pair_moves[(move.src_sp_rank, move.dst_sp_rank)].append(move)

        sent_bytes = 0
        received_bytes = 0
        local_chunks = 0
        local_moves = 0
        role = "idle"

        for src_rank, dst_rank in sorted(pair_moves):
            for chunk in self._chunks(pair_moves[(src_rank, dst_rank)]):
                if sp_rank not in {src_rank, dst_rank}:
                    continue

                token_count = sum(move.num_tokens for move in chunk)
                buffer = self.scratch[:token_count]
                if not buffer.is_contiguous():
                    raise RuntimeError("migration scratch view must be contiguous")

                peer_group_rank = dst_rank if sp_rank == src_rank else src_rank
                peer_global_rank = dist.get_global_rank(self.group, peer_group_rank)

                if sp_rank == src_rank:
                    role = "source"
                    self._pack(chunk, buffer)
                    work = dist.isend(buffer, dst=peer_global_rank, group=self.group)
                    if work is None:
                        raise RuntimeError("dist.isend returned no Work handle")
                    work.wait()
                    sent_bytes += token_count * self.token_bytes
                else:
                    role = "destination"
                    work = dist.irecv(buffer, src=peer_global_rank, group=self.group)
                    if work is None:
                        raise RuntimeError("dist.irecv returned no Work handle")
                    work.wait()
                    self._scatter(chunk, buffer)
                    received_bytes += token_count * self.token_bytes

                local_chunks += 1
                local_moves += len(chunk)

        return KVCacheP2PResult(
            dp_idx=current_dp_idx,
            sp_rank=sp_rank,
            role=role,
            num_moves=local_moves,
            num_chunks=local_chunks,
            sent_bytes=sent_bytes,
            received_bytes=received_bytes,
        )

    def _validate_and_normalize(
        self, moves: Sequence[KVCacheP2PMove]
    ) -> tuple[KVCacheP2PMove, ...]:
        if not moves:
            return ()
        if not all(isinstance(move, KVCacheP2PMove) for move in moves):
            raise TypeError("moves must contain KVCacheP2PMove objects")

        world_size = dist.get_world_size(self.group)
        dp_indices = {move.dp_idx for move in moves}
        source_ranks = {move.src_sp_rank for move in moves}
        if len(dp_indices) != 1:
            raise ValueError("one P2P plan must target exactly one DP")
        if len(source_ranks) != 1:
            raise ValueError("one P2P plan must evacuate exactly one source rank")

        num_blocks = int(self.kv_cache.size(2))
        for move in moves:
            if move.dp_idx < 0:
                raise ValueError("dp_idx must be non-negative")
            if not 0 <= move.src_sp_rank < world_size:
                raise ValueError("source SP rank is outside attn_sp_group")
            if not 0 <= move.dst_sp_rank < world_size:
                raise ValueError("destination SP rank is outside attn_sp_group")
            if move.src_sp_rank == move.dst_sp_rank:
                raise ValueError("source and destination SP ranks must differ")
            if not 0 <= move.src_block_id < num_blocks:
                raise ValueError("source block ID is outside the local KV cache")
            if not 0 <= move.dst_block_id < num_blocks:
                raise ValueError("destination block ID is outside the local KV cache")
            if move.num_tokens <= 0:
                raise ValueError("num_tokens must be > 0")
            if (
                move.src_token_offset < 0
                or move.src_token_offset + move.num_tokens > self.block_size
            ):
                raise ValueError("source token range crosses a KV block boundary")
            if (
                move.dst_token_offset < 0
                or move.dst_token_offset + move.num_tokens > self.block_size
            ):
                raise ValueError("destination token range crosses a KV block boundary")

        self._validate_non_overlapping(moves, source=True)
        self._validate_non_overlapping(moves, source=False)
        return tuple(
            sorted(
                moves,
                key=lambda move: (
                    move.dp_idx,
                    move.src_sp_rank,
                    move.dst_sp_rank,
                    move.src_block_id,
                    move.src_token_offset,
                    move.dst_block_id,
                    move.dst_token_offset,
                ),
            )
        )

    @staticmethod
    def _validate_non_overlapping(
        moves: Sequence[KVCacheP2PMove], *, source: bool
    ) -> None:
        ranges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for move in moves:
            if source:
                key = (move.src_sp_rank, move.src_block_id)
                begin = move.src_token_offset
            else:
                key = (move.dst_sp_rank, move.dst_block_id)
                begin = move.dst_token_offset
            ranges[key].append((begin, begin + move.num_tokens))

        side = "source" if source else "destination"
        for intervals in ranges.values():
            intervals.sort()
            for previous, current in zip(intervals, intervals[1:]):
                if current[0] < previous[1]:
                    raise ValueError(f"overlapping {side} token ranges")

    def _split_large_moves(
        self, moves: Iterable[KVCacheP2PMove]
    ) -> Iterable[KVCacheP2PMove]:
        for move in moves:
            consumed = 0
            while consumed < move.num_tokens:
                length = min(self.chunk_tokens, move.num_tokens - consumed)
                yield dataclasses.replace(
                    move,
                    src_token_offset=move.src_token_offset + consumed,
                    dst_token_offset=move.dst_token_offset + consumed,
                    num_tokens=length,
                )
                consumed += length

    def _chunks(
        self, moves: Sequence[KVCacheP2PMove]
    ) -> Iterable[tuple[KVCacheP2PMove, ...]]:
        chunk: list[KVCacheP2PMove] = []
        chunk_tokens = 0
        for move in moves:
            if chunk and chunk_tokens + move.num_tokens > self.chunk_tokens:
                yield tuple(chunk)
                chunk = []
                chunk_tokens = 0
            chunk.append(move)
            chunk_tokens += move.num_tokens
        if chunk:
            yield tuple(chunk)

    def _pack(self, moves: Sequence[KVCacheP2PMove], buffer: torch.Tensor) -> None:
        cursor = 0
        for move in moves:
            length = move.num_tokens
            source = self.kv_cache[
                :,
                :,
                move.src_block_id,
                move.src_token_offset : move.src_token_offset + length,
                :,
                :,
            ]
            buffer[cursor : cursor + length].copy_(source.permute(2, 0, 1, 3, 4))
            cursor += length

    def _scatter(self, moves: Sequence[KVCacheP2PMove], buffer: torch.Tensor) -> None:
        cursor = 0
        for move in moves:
            length = move.num_tokens
            destination = self.kv_cache[
                :,
                :,
                move.dst_block_id,
                move.dst_token_offset : move.dst_token_offset + length,
                :,
                :,
            ]
            destination.copy_(buffer[cursor : cursor + length].permute(1, 2, 0, 3, 4))
            cursor += length
