from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


HIERARCHICAL_LOOP_COUNT = 16
CONTROL_DUMMY_SCHEMA_VERSION = 2
UINT64_MAX = (1 << 64) - 1


def round_up(value: int, quantum: int = HIERARCHICAL_LOOP_COUNT) -> int:
    if value < 0:
        raise ValueError("value must be non-negative")
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    return ((value + quantum - 1) // quantum) * quantum


class RequestState(str, Enum):
    PENDING_ADD = "PENDING_ADD"
    WAITING_ADMISSION = "WAITING_ADMISSION"
    RUNNING_DECODE = "RUNNING_DECODE"
    ABORT_PENDING = "ABORT_PENDING"
    FINISHED = "FINISHED"
    ABORTED = "ABORTED"
    REJECTED = "REJECTED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            RequestState.FINISHED,
            RequestState.ABORTED,
            RequestState.REJECTED,
        }


class OwnerState(str, Enum):
    PENDING_OWNER = "PENDING_OWNER"
    OWNED = "OWNED"


@dataclass(frozen=True, slots=True)
class RequestValidation:
    original_prompt_len: int
    internal_prompt_len: int
    padded_completion_len: int
    total_capacity_len: int


def validate_add_request(
    *,
    request_id: int,
    prompt_token_ids: Iterable[int],
    max_tokens: int,
    ignore_eos: bool,
    max_model_len: int,
    vocab_size: int,
) -> RequestValidation:
    if not 0 <= request_id <= UINT64_MAX:
        raise ValueError("request_id must fit uint64")
    prompt = tuple(prompt_token_ids)
    if not prompt:
        raise ValueError("prompt_token_ids must not be empty")
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if not ignore_eos:
        raise ValueError("hierarchical scheduler requires ignore_eos=True")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    invalid_token = next(
        (token_id for token_id in prompt if not 0 <= token_id < vocab_size),
        None,
    )
    if invalid_token is not None:
        raise ValueError(
            f"prompt token id {invalid_token} is outside [0, {vocab_size})"
        )

    padded_completion_len = round_up(max_tokens)
    total_capacity_len = len(prompt) + 1 + padded_completion_len
    if total_capacity_len > max_model_len:
        raise ValueError(
            "request exceeds hierarchical padded model length: "
            f"prompt={len(prompt)} + bootstrap=1 + "
            f"padded_completion={padded_completion_len} > "
            f"max_model_len={max_model_len}"
        )
    return RequestValidation(
        original_prompt_len=len(prompt),
        internal_prompt_len=len(prompt) + 1,
        padded_completion_len=padded_completion_len,
        total_capacity_len=total_capacity_len,
    )


@dataclass(frozen=True, slots=True)
class AddCommand:
    request_id: int
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    temperature: float
    ignore_eos: bool
    wave_id: int


@dataclass(frozen=True, slots=True)
class AddResult:
    request_id: int
    accepted: bool
    engine_id: int | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class AbortResult:
    request_id: int
    status: str


@dataclass(frozen=True, slots=True)
class FinishEvent:
    request_id: int
    generated_count: int
    status: str
    engine_id: int


@dataclass(frozen=True, slots=True)
class LoadSnapshot:
    engine_id: int
    ready: bool
    waiting: int
    running: int
    free_blocks_min: int
    wave_id: int
    quantum_id: int
    useful_real_batch_size: int = 0
    control_dummy_count: int = 0
    all_dummy_engine_quantums: int = 0


@dataclass(frozen=True, slots=True)
class WorkerDecodeResult:
    wave_id: int
    quantum_id: int
    global_rank: int
    forward_count: int
    mastered_request_ids: tuple[int, ...]
    sampled_token_ids: tuple[tuple[int, ...], ...]

    def validate_quantum(self, wave_id: int, quantum_id: int) -> None:
        if (self.wave_id, self.quantum_id) != (wave_id, quantum_id):
            raise ValueError(
                "worker result step mismatch: "
                f"expected ({wave_id}, {quantum_id}), got "
                f"({self.wave_id}, {self.quantum_id})"
            )
        if self.forward_count != HIERARCHICAL_LOOP_COUNT:
            raise ValueError(
                "hierarchical worker must execute exactly "
                f"{HIERARCHICAL_LOOP_COUNT} forwards, got {self.forward_count}"
            )
        if len(self.mastered_request_ids) != len(self.sampled_token_ids):
            raise ValueError(
                "mastered_request_ids and sampled_token_ids length mismatch"
            )
        if any(
            len(tokens) != HIERARCHICAL_LOOP_COUNT
            for tokens in self.sampled_token_ids
        ):
            raise ValueError(
                "each mastered request must return exactly "
                f"{HIERARCHICAL_LOOP_COUNT} sampled token ids"
            )


@dataclass(slots=True)
class LocalDecodeBatch:
    wave_id: int
    quantum_id: int
    engine_id: int
    engine_has_real: bool
    per_rank_sequences: dict[int, list[Any]]
    request_master_global_rank: dict[int, int]
    frozen_request_order: dict[int, tuple[int, ...]]
    control_dummy_ids: frozenset[int]
    _all_sequences: list[Any] = field(repr=False, default_factory=list)

    def expected_request_ids(self, global_rank: int) -> tuple[int, ...]:
        return self.frozen_request_order.get(global_rank, ())

    def validate_worker_results(
        self, results: Iterable[WorkerDecodeResult]
    ) -> dict[int, WorkerDecodeResult]:
        by_rank: dict[int, WorkerDecodeResult] = {}
        for result in results:
            result.validate_quantum(self.wave_id, self.quantum_id)
            if result.global_rank in by_rank:
                raise ValueError(
                    f"duplicate worker result for global rank {result.global_rank}"
                )
            expected = self.expected_request_ids(result.global_rank)
            if result.mastered_request_ids != expected:
                raise ValueError(
                    f"worker rank {result.global_rank} request order mismatch: "
                    f"expected {expected}, got {result.mastered_request_ids}"
                )
            by_rank[result.global_rank] = result

        expected_ranks = set(self.per_rank_sequences)
        missing = expected_ranks.difference(by_rank)
        extra = set(by_rank).difference(expected_ranks)
        if missing or extra:
            raise ValueError(
                f"worker result rank mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        return by_rank
