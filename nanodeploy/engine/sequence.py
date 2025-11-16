import dataclasses
import uuid
from collections import defaultdict
from copy import copy
from enum import auto, Enum
from itertools import count

from pydantic import BaseModel

from nanodeploy.metrics import SeqMetrics

from nanodeploy.sampling_params import SamplingParams


class SequenceConfig(BaseModel):
    temperature: float | None = None


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

    TO_BE_MIGRATED = auto()


@dataclasses.dataclass
class BlockContext:
    engine_id: str | None

    dp_idx: int
    master_sp_idx: int

    attention_sp: int
    attention_dp: int

    # For Sequence Parallelization
    block_location: list[tuple[int, int]]
    num_dispatched_tokens: dict[int, int]
    sp_block_table: dict[int, list[int]]


class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        engine_id: str | None = None,
        master_sp_rank: int = 0,
    ):
        sampling_params = sampling_params or SamplingParams()
        self.seq_id = str(uuid.uuid4())
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)

        self.num_prompt_tokens = len(token_ids)

        # total tokens since preemption happens
        self.num_checkpointed_tokens = len(token_ids)
        self.num_cached_tokens = 0

        self.backup_engine_id: str | None = engine_id
        self.active_engine_id: str | None = engine_id
        self.block_ctx_map: dict[str | None, BlockContext] = {
            engine_id: BlockContext(
                engine_id=engine_id,
                dp_idx=-1,
                attention_sp=1,
                attention_dp=1,
                master_sp_idx=master_sp_rank,
                block_location=[],
                sp_block_table=defaultdict(list),
                num_dispatched_tokens=defaultdict(int),
            )
        }

        self.metrics = SeqMetrics()

        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def dp_idx(self, engine_id):
        return self.block_ctx_map[engine_id].dp_idx

    def block_ctx(self, engine_id: str | None = None):
        engine_id = engine_id or self.active_engine_id
        return self.block_ctx_map[engine_id]

    def block_table(self, engine_id: str | None = None, sp_idx: int = 0):
        engine_id = engine_id or self.active_engine_id
        return self.block_ctx(engine_id).sp_block_table[sp_idx]

    def set_engine_id(self, engine_id: str, attention_dp=1, attention_sp: int = 1):
        self.active_engine_id = engine_id
        if engine_id in self.block_ctx_map:
            return
        self.block_ctx_map[engine_id] = BlockContext(
            engine_id=engine_id,
            dp_idx=-1,
            master_sp_idx=0,
            attention_sp=attention_sp,
            attention_dp=attention_dp,
            block_location=[],
            sp_block_table=defaultdict(list, {i: [] for i in range(attention_sp)}),
            num_dispatched_tokens=defaultdict(int),
        )

    def context_len(self, engine_id: str | None = None, sp_idx: int | None = None):
        sp_idx = (
            sp_idx if sp_idx is not None else self.block_ctx(engine_id).master_sp_idx
        )
        return self.block_ctx(engine_id).num_dispatched_tokens[sp_idx]

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completed_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_generated_tokens_since_checkpoint(self):
        return self.num_tokens - self.num_checkpointed_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    def num_blocks(self, engine_id: str | None, sp_idx: int):
        return (
            self.block_ctx(engine_id).num_dispatched_tokens[sp_idx]
            + self.block_size
            - 1
        ) // self.block_size

    def last_block_page_id(self, engine_id: str | None, sp_idx: int):
        num_tokens = self.block_ctx(engine_id).num_dispatched_tokens[sp_idx]
        last_block_idx = (num_tokens - 1) // self.block_size
        return self.block_table(engine_id, sp_idx)[last_block_idx]

    def last_block_num_tokens(self, engine_id: str | None, sp_idx: int):
        num_tokens = self.block_ctx(engine_id).num_dispatched_tokens[sp_idx]
        return num_tokens - (self.num_blocks(engine_id, sp_idx) - 1) * self.block_size

    def block(self, i, engine_id: str | None, sp_idx: int):
        assert 0 <= i < self.num_blocks(engine_id, sp_idx)
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def append_token(
        self, token_id: int, engine_id: str | None = None, sp_idx: int | None = None
    ):
        engine_id = engine_id or self.active_engine_id
        sp_idx = (
            sp_idx if sp_idx is not None else self.block_ctx(engine_id).master_sp_idx
        )
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        self.block_ctx(engine_id).num_dispatched_tokens[sp_idx] += 1

    def __getstate__(self):
        return (
            self.num_tokens,
            self.num_checkpointed_tokens,
            self.num_cached_tokens,
            self.backup_engine_id,
            self.active_engine_id,
            self.block_ctx_map,
            self.temperature,
            (
                self.token_ids
                if self.num_generated_tokens_since_checkpoint == 0
                else self.last_token
            ),
        )

    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_checkpointed_tokens,
            self.num_cached_tokens,
            self.backup_engine_id,
            self.active_engine_id,
            self.block_ctx_map,
            self.temperature,
        ) = state[:-1]
        if self.num_generated_tokens_since_checkpoint == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
