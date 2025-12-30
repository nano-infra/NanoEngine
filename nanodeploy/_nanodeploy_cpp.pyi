"""
NanoDeploy C++ Backend
"""

from __future__ import annotations

import collections.abc
import typing

__all__: list[str] = [
    "ACTIVE",
    "Block",
    "BlockContext",
    "BlockContextSlot",
    "BlockIdList",
    "BlockLocationList",
    "BlockManager",
    "BlockManagerMap",
    "DecodeMetadata",
    "DefaultIntDict",
    "DefaultListDict",
    "FINISHED",
    "LeastBatch",
    "LeastCache",
    "MIGRATE",
    "MigrationMap",
    "PrefillMetadata",
    "RUNNING",
    "RoundRobin",
    "RoutingStrategy",
    "SPStateManager",
    "SPStateManagerList",
    "SWAP",
    "ScheduleResult",
    "Scheduler",
    "Sequence",
    "SequenceDeque",
    "SequenceMetric",
    "SequenceStatus",
    "ServerMetric",
    "TO_BE_MIGRATED",
    "WAITING",
    "deserialize",
    "postprocess_sequences",
    "prepare_decode_cpp",
    "prepare_prefill_cpp",
    "serialize",
    "update_seqs_inner_loop",
]

class Block:
    def __init__(self, arg0: typing.SupportsInt) -> None: ...
    def reset(self) -> None: ...
    def update(
        self,
        arg0: typing.SupportsInt,
        arg1: collections.abc.Sequence[typing.SupportsInt],
    ) -> None: ...
    @property
    def block_id(self) -> int: ...
    @block_id.setter
    def block_id(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def hash(self) -> int: ...
    @hash.setter
    def hash(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def ref_count(self) -> int: ...
    @ref_count.setter
    def ref_count(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def token_ids(self) -> list[int]: ...
    @token_ids.setter
    def token_ids(self, arg0: collections.abc.Sequence[typing.SupportsInt]) -> None: ...

class BlockContext:
    block_location: BlockLocationList
    engine_id: str
    sp_block_table: DefaultListDict
    def __getstate__(
        self,
    ) -> tuple[
        str, int, int, int, int, list[tuple[int, int]], list[list[int]], list[int]
    ]: ...
    def __init__(self) -> None: ...
    def __setstate__(
        self,
        arg0: tuple[
            str,
            typing.SupportsInt,
            typing.SupportsInt,
            typing.SupportsInt,
            typing.SupportsInt,
            collections.abc.Sequence[tuple[typing.SupportsInt, typing.SupportsInt]],
            collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]],
            collections.abc.Sequence[typing.SupportsInt],
        ],
    ) -> None: ...
    def reset(
        self,
        engine_id: str,
        attention_sp: typing.SupportsInt,
        attention_dp: typing.SupportsInt,
    ) -> None: ...
    @property
    def attention_dp(self) -> int: ...
    @attention_dp.setter
    def attention_dp(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def attention_sp(self) -> int: ...
    @attention_sp.setter
    def attention_sp(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def dp_idx(self) -> int: ...
    @dp_idx.setter
    def dp_idx(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def master_sp_idx(self) -> int: ...
    @master_sp_idx.setter
    def master_sp_idx(self, arg0: typing.SupportsInt) -> None: ...

class BlockContextSlot:
    """
    Members:

      ACTIVE

      MIGRATE

      SWAP
    """

    ACTIVE: typing.ClassVar[BlockContextSlot]  # value = <BlockContextSlot.ACTIVE: 0>
    MIGRATE: typing.ClassVar[BlockContextSlot]  # value = <BlockContextSlot.MIGRATE: 1>
    SWAP: typing.ClassVar[BlockContextSlot]  # value = <BlockContextSlot.SWAP: 2>
    __members__: typing.ClassVar[
        dict[str, BlockContextSlot]
    ]  # value = {'ACTIVE': <BlockContextSlot.ACTIVE: 0>, 'MIGRATE': <BlockContextSlot.MIGRATE: 1>, 'SWAP': <BlockContextSlot.SWAP: 2>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class BlockIdList:
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """

    def __contains__(self, x: typing.SupportsInt) -> bool:
        """
        Return true the container contains ``x``
        """

    @typing.overload
    def __delitem__(self, arg0: typing.SupportsInt) -> None:
        """
        Delete the list elements at index ``i``
        """

    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """

    def __eq__(self, arg0: BlockIdList) -> bool: ...
    @typing.overload
    def __getitem__(self, s: slice) -> BlockIdList:
        """
        Retrieve list elements using a slice object
        """

    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: BlockIdList) -> None:
        """
        Copy constructor
        """

    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None: ...
    def __iter__(self) -> collections.abc.Iterator[int]: ...
    def __len__(self) -> int: ...
    def __ne__(self, arg0: BlockIdList) -> bool: ...
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """

    @typing.overload
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: BlockIdList) -> None:
        """
        Assign list elements using a slice object
        """

    def append(self, x: typing.SupportsInt) -> None:
        """
        Add an item to the end of the list
        """

    def clear(self) -> None:
        """
        Clear the contents
        """

    def count(self, x: typing.SupportsInt) -> int:
        """
        Return the number of times ``x`` appears in the list
        """

    @typing.overload
    def extend(self, L: BlockIdList) -> None:
        """
        Extend the list by appending all the items in the given list
        """

    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """

    def insert(self, i: typing.SupportsInt, x: typing.SupportsInt) -> None:
        """
        Insert an item at a given position.
        """

    @typing.overload
    def pop(self) -> int:
        """
        Remove and return the last item
        """

    @typing.overload
    def pop(self, i: typing.SupportsInt) -> int:
        """
        Remove and return the item at index ``i``
        """

    def remove(self, x: typing.SupportsInt) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """

class BlockLocationList:
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """

    def __contains__(self, x: tuple[typing.SupportsInt, typing.SupportsInt]) -> bool:
        """
        Return true the container contains ``x``
        """

    @typing.overload
    def __delitem__(self, arg0: typing.SupportsInt) -> None:
        """
        Delete the list elements at index ``i``
        """

    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """

    def __eq__(self, arg0: BlockLocationList) -> bool: ...
    @typing.overload
    def __getitem__(self, s: slice) -> BlockLocationList:
        """
        Retrieve list elements using a slice object
        """

    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> tuple[int, int]: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: BlockLocationList) -> None:
        """
        Copy constructor
        """

    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    def __iter__(self) -> collections.abc.Iterator[tuple[int, int]]: ...
    def __len__(self) -> int: ...
    def __ne__(self, arg0: BlockLocationList) -> bool: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.SupportsInt,
        arg1: tuple[typing.SupportsInt, typing.SupportsInt],
    ) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: BlockLocationList) -> None:
        """
        Assign list elements using a slice object
        """

    def append(self, x: tuple[typing.SupportsInt, typing.SupportsInt]) -> None:
        """
        Add an item to the end of the list
        """

    def clear(self) -> None:
        """
        Clear the contents
        """

    def count(self, x: tuple[typing.SupportsInt, typing.SupportsInt]) -> int:
        """
        Return the number of times ``x`` appears in the list
        """

    @typing.overload
    def extend(self, L: BlockLocationList) -> None:
        """
        Extend the list by appending all the items in the given list
        """

    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """

    def insert(
        self, i: typing.SupportsInt, x: tuple[typing.SupportsInt, typing.SupportsInt]
    ) -> None:
        """
        Insert an item at a given position.
        """

    @typing.overload
    def pop(self) -> tuple[int, int]:
        """
        Remove and return the last item
        """

    @typing.overload
    def pop(self, i: typing.SupportsInt) -> tuple[int, int]:
        """
        Remove and return the item at index ``i``
        """

    def remove(self, x: tuple[typing.SupportsInt, typing.SupportsInt]) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """

class BlockManager:
    @staticmethod
    def compute_hash(
        token_ids: collections.abc.Sequence[typing.SupportsInt],
        prefix: typing.SupportsInt = -1,
    ) -> int: ...
    def __init__(
        self,
        engine_id: str,
        sp_idx: typing.SupportsInt,
        num_blocks: typing.SupportsInt,
        block_size: typing.SupportsInt,
    ) -> None: ...
    def allocate(
        self,
        seq: typing.Sequence,
        token_idx_from: typing.SupportsInt = -1,
        token_idx_to: typing.SupportsInt = -1,
    ) -> None: ...
    def can_allocate(self, arg0: typing.Sequence) -> bool: ...
    def can_append(
        self, seq: typing.Sequence, num_tokens: typing.SupportsInt = 1
    ) -> bool: ...
    def deallocate(self, arg0: typing.Sequence, arg1: BlockContextSlot) -> None: ...
    def may_append(
        self, seq: typing.Sequence, num_tokens: typing.SupportsInt = 1
    ) -> bool: ...
    @property
    def blocks(self) -> list[Block]: ...
    @property
    def free_block_ids(self) -> list[int]: ...
    @property
    def num_free_blocks(self) -> int: ...

class BlockManagerMap:
    def __bool__(self) -> bool:
        """
        Check whether the map is nonempty
        """

    @typing.overload
    def __contains__(self, arg0: typing.SupportsInt) -> bool: ...
    @typing.overload
    def __contains__(self, arg0: typing.Any) -> bool: ...
    def __delitem__(self, arg0: typing.SupportsInt) -> None: ...
    def __getitem__(self, arg0: typing.SupportsInt) -> BlockManager: ...
    def __init__(self) -> None: ...
    def __iter__(self) -> collections.abc.Iterator[int]: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this map.
        """

    def __setitem__(self, arg0: typing.SupportsInt, arg1: BlockManager) -> None: ...
    def items(self) -> typing.ItemsView: ...
    def keys(self) -> typing.KeysView: ...
    def values(self) -> typing.ValuesView: ...

class DecodeMetadata:
    @property
    def block_tables_flat(self) -> list[int]: ...
    @property
    def context_lens_flat(self) -> list[int]: ...
    @property
    def global_context_lens_flat(self) -> list[int]: ...
    @property
    def input_ids(self) -> list[int]: ...
    @property
    def max_num_blocks(self) -> int: ...
    @property
    def positions(self) -> list[int]: ...
    @property
    def slot_mapping(self) -> list[int]: ...

class DefaultIntDict:
    def __contains__(self, arg0: typing.SupportsInt) -> bool: ...
    def __getitem__(self, arg0: typing.SupportsInt) -> int: ...
    def __getstate__(self) -> dict: ...
    def __init__(self) -> None: ...
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    def __setstate__(self, arg0: dict) -> None: ...
    def clear(self) -> None: ...
    def items(self) -> list: ...
    def keys(self) -> list: ...

class DefaultListDict:
    def __getitem__(self, arg0: typing.SupportsInt) -> BlockIdList: ...
    def __init__(self) -> None: ...
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: collections.abc.Iterable
    ) -> None: ...

class MigrationMap:
    def __contains__(self, arg0: str) -> bool: ...
    def __delitem__(self, arg0: str) -> None: ...
    def __getitem__(self, arg0: str) -> tuple[typing.Sequence, int]: ...
    def __init__(self) -> None: ...
    def __len__(self) -> int: ...
    def __setitem__(
        self, arg0: str, arg1: tuple[typing.Sequence, typing.SupportsInt]
    ) -> None: ...
    def items(self) -> list: ...
    def keys(self) -> list: ...

class PrefillMetadata:
    @property
    def block_tables_flat(self) -> list[int]: ...
    @property
    def cu_seqlens_k(self) -> list[int]: ...
    @property
    def cu_seqlens_q(self) -> list[int]: ...
    @property
    def input_ids(self) -> list[int]: ...
    @property
    def max_num_blocks(self) -> int: ...
    @property
    def max_seqlen_k(self) -> int: ...
    @property
    def max_seqlen_q(self) -> int: ...
    @property
    def positions(self) -> list[int]: ...
    @property
    def slot_mapping(self) -> list[int]: ...
    @property
    def use_block_tables(self) -> bool: ...

class RoutingStrategy:
    """
    Members:

      RoundRobin

      LeastBatch

      LeastCache
    """

    LeastBatch: typing.ClassVar[
        RoutingStrategy
    ]  # value = <RoutingStrategy.LeastBatch: 1>
    LeastCache: typing.ClassVar[
        RoutingStrategy
    ]  # value = <RoutingStrategy.LeastCache: 2>
    RoundRobin: typing.ClassVar[
        RoutingStrategy
    ]  # value = <RoutingStrategy.RoundRobin: 0>
    __members__: typing.ClassVar[
        dict[str, RoutingStrategy]
    ]  # value = {'RoundRobin': <RoutingStrategy.RoundRobin: 0>, 'LeastBatch': <RoutingStrategy.LeastBatch: 1>, 'LeastCache': <RoutingStrategy.LeastCache: 2>}
    @staticmethod
    def __class_getitem__(arg0: str) -> RoutingStrategy: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SPStateManager:
    block_manager: BlockManagerMap
    routing_strategy: RoutingStrategy
    running: SequenceDeque
    def __init__(
        self,
        engine_id: str,
        attention_sp: typing.SupportsInt,
        num_kvcache_blocks: typing.SupportsInt,
        kvcache_block_size: typing.SupportsInt,
        max_num_seqs: typing.SupportsInt,
        max_num_batched_tokens: typing.SupportsInt,
    ) -> None: ...
    def allocate(self, seq: typing.Sequence) -> None: ...
    def can_allocate(
        self,
        seq: typing.Sequence,
        num_seqs: DefaultIntDict,
        num_batched_tokens: DefaultIntDict,
    ) -> bool: ...
    def can_append(
        self, seq: typing.Sequence, num_tokens: typing.SupportsInt = 1
    ) -> bool: ...
    def deallocate(self, seq: typing.Sequence, slot: BlockContextSlot) -> None: ...
    def may_append(
        self, seq: typing.Sequence, num_tokens: typing.SupportsInt = 1
    ) -> bool: ...
    @property
    def dummy_seqs(self) -> list[typing.Sequence]: ...
    @dummy_seqs.setter
    def dummy_seqs(self, arg0: collections.abc.Sequence[typing.Sequence]) -> None: ...
    @property
    def is_empty(self) -> bool: ...

class SPStateManagerList:
    def __getitem__(self, arg0: typing.SupportsInt) -> SPStateManager: ...
    def __init__(self) -> None: ...
    def __iter__(self) -> collections.abc.Iterator[SPStateManager]: ...
    def __len__(self) -> int: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: SPStateManager) -> None: ...

class ScheduleResult:
    is_prefill: bool
    @property
    def dp_seqs(self) -> list[list[typing.Sequence]]: ...
    @dp_seqs.setter
    def dp_seqs(
        self, arg0: collections.abc.Sequence[collections.abc.Sequence[typing.Sequence]]
    ) -> None: ...
    @property
    def dp_sp_seqs(self) -> list[list[typing.Sequence]]: ...
    @dp_sp_seqs.setter
    def dp_sp_seqs(
        self, arg0: collections.abc.Sequence[collections.abc.Sequence[typing.Sequence]]
    ) -> None: ...
    @property
    def filtered_dp_sp_seqs(self) -> list[list[typing.Sequence]]: ...
    @filtered_dp_sp_seqs.setter
    def filtered_dp_sp_seqs(
        self, arg0: collections.abc.Sequence[collections.abc.Sequence[typing.Sequence]]
    ) -> None: ...

class Scheduler:
    routing_strategy: RoutingStrategy
    waiting: SequenceDeque
    waiting_migration: SequenceDeque
    worker_state: SPStateManagerList
    def __init__(
        self,
        engine_id: str,
        loop_count: typing.SupportsInt,
        max_num_seqs: typing.SupportsInt,
        max_num_batched_tokens: typing.SupportsInt,
        eos: typing.SupportsInt,
        attention_dp: typing.SupportsInt,
        attention_sp: typing.SupportsInt,
        num_kvcache_blocks: typing.SupportsInt,
        kvcache_block_size: typing.SupportsInt,
        mode: str,
    ) -> None: ...
    def add(self, seq: typing.Sequence) -> None: ...
    def block_manager(self, dp_idx: typing.SupportsInt) -> dict[int, BlockManager]: ...
    @typing.overload
    def free_to_be_migrated(self, seq: typing.Sequence) -> None: ...
    @typing.overload
    def free_to_be_migrated(
        self, seqs: collections.abc.Sequence[typing.Sequence]
    ) -> None: ...
    def is_finished(self) -> bool: ...
    def postprocess(
        self,
        dp_seqs: collections.abc.Sequence[collections.abc.Sequence[typing.Sequence]],
        dp_token_ids: collections.abc.Sequence[
            collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]]
        ],
        update_metrics: bool = True,
    ) -> None: ...
    def preempt(self, dp_idx: typing.SupportsInt, seq: typing.Sequence) -> None: ...
    def running(self, dp_idx: typing.SupportsInt) -> SequenceDeque: ...
    def schedule(self) -> ScheduleResult: ...
    @property
    def to_be_migrated(self) -> dict[int, tuple[typing.Sequence, int]]: ...
    @to_be_migrated.setter
    def to_be_migrated(
        self,
        arg0: collections.abc.Mapping[
            typing.SupportsInt, tuple[typing.Sequence, typing.SupportsInt]
        ],
    ) -> None: ...

class Sequence:
    block_size: typing.ClassVar[int] = 256
    ignore_eos: bool
    metric: SequenceMetric
    status: SequenceStatus
    def __getitem__(self: typing.Sequence, arg0: typing.Any) -> typing.Any: ...
    def __getstate__(
        self: typing.Sequence,
    ) -> tuple[
        int,
        int,
        int,
        typing.Annotated[list[BlockContext], "FixedSize(3)"],
        float,
        list[int],
    ]: ...
    def __init__(
        self: typing.Sequence,
        token_ids: collections.abc.Sequence[typing.SupportsInt],
        temperature: typing.SupportsFloat = 1.0,
        max_tokens: typing.SupportsInt = 256,
        ignore_eos: bool = False,
    ) -> None: ...
    def __len__(self: typing.Sequence) -> int: ...
    def __setstate__(
        self: typing.Sequence,
        arg0: tuple[
            typing.SupportsInt,
            typing.SupportsInt,
            typing.SupportsInt,
            typing.Annotated[collections.abc.Sequence[BlockContext], "FixedSize(3)"],
            typing.SupportsFloat,
            collections.abc.Sequence[typing.SupportsInt],
        ],
    ) -> None: ...
    def active(
        self: typing.Sequence,
        engine_id: str,
        attention_sp: typing.SupportsInt,
        attention_dp: typing.SupportsInt,
    ) -> int: ...
    def append_token(
        self: typing.Sequence,
        token_id: typing.SupportsInt,
        slot: BlockContextSlot,
        sp_idx: typing.SupportsInt | None = None,
    ) -> None: ...
    def block(
        self: typing.Sequence,
        i: typing.SupportsInt,
        slot: BlockContextSlot,
        sp_idx: typing.SupportsInt,
    ) -> list[int]: ...
    def block_ctx(
        self: typing.Sequence, slot: BlockContextSlot = ...
    ) -> BlockContext: ...
    def block_table(
        self: typing.Sequence, slot: BlockContextSlot, sp_idx: typing.SupportsInt = 0
    ) -> BlockIdList: ...
    def context_len(
        self: typing.Sequence,
        engine_id: BlockContextSlot,
        sp_idx: typing.SupportsInt | None = None,
    ) -> int: ...
    def dp_idx(self: typing.Sequence, slot: BlockContextSlot) -> int: ...
    def last_block_num_tokens(
        self: typing.Sequence, slot: BlockContextSlot, sp_idx: typing.SupportsInt
    ) -> int: ...
    def last_block_page_id(
        self: typing.Sequence, slot: BlockContextSlot, sp_idx: typing.SupportsInt
    ) -> int: ...
    def migrate(self: typing.Sequence) -> int: ...
    def num_blocks(
        self: typing.Sequence, slot: BlockContextSlot, sp_idx: typing.SupportsInt
    ) -> int: ...
    @property
    def completion_token_ids(self) -> list[int]: ...
    @property
    def is_finished(self) -> bool: ...
    @property
    def last_token(self) -> int: ...
    @last_token.setter
    def last_token(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def max_tokens(self) -> int: ...
    @max_tokens.setter
    def max_tokens(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def num_cached_blocks(self) -> int: ...
    @property
    def num_cached_tokens(self) -> int: ...
    @num_cached_tokens.setter
    def num_cached_tokens(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def num_checkpointed_tokens(self) -> int: ...
    @num_checkpointed_tokens.setter
    def num_checkpointed_tokens(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def num_completed_tokens(self) -> int: ...
    @property
    def num_generated_tokens_since_checkpoint(self) -> int: ...
    @property
    def num_prompt_tokens(self) -> int: ...
    @num_prompt_tokens.setter
    def num_prompt_tokens(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def num_tokens(self) -> int: ...
    @num_tokens.setter
    def num_tokens(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def prompt_token_ids(self) -> list[int]: ...
    @property
    def seq_id(self) -> int: ...
    @seq_id.setter
    def seq_id(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def temperature(self) -> float: ...
    @temperature.setter
    def temperature(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def token_ids(self) -> list[int]: ...
    @token_ids.setter
    def token_ids(self, arg0: collections.abc.Sequence[typing.SupportsInt]) -> None: ...

class SequenceDeque:
    def __bool__(self) -> bool: ...
    def __getitem__(self, arg0: typing.SupportsInt) -> typing.Sequence: ...
    def __init__(self) -> None: ...
    def __iter__(self) -> collections.abc.Iterator[typing.Sequence]: ...
    def __len__(self) -> int: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: typing.Sequence) -> None: ...
    def append(self, arg0: typing.Sequence) -> None: ...
    def extendleft(self, arg0: typing.Any) -> None: ...
    def pop(self) -> typing.Sequence: ...
    def popleft(self) -> typing.Sequence: ...
    def remove(self, arg0: typing.Sequence) -> None: ...

class SequenceMetric:
    def __getstate__(
        self,
    ) -> tuple[
        int,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        int,
        int,
        list[float],
    ]: ...
    def __init__(
        self, seq_id: typing.SupportsInt, num_prompt_tokens: typing.SupportsInt = 0
    ) -> None: ...
    def __setstate__(
        self,
        arg0: tuple[
            typing.SupportsInt,
            typing.SupportsFloat | None,
            typing.SupportsFloat | None,
            typing.SupportsFloat | None,
            typing.SupportsFloat | None,
            typing.SupportsFloat | None,
            typing.SupportsFloat | None,
            typing.SupportsFloat | None,
            typing.SupportsInt,
            typing.SupportsInt,
            collections.abc.Sequence[typing.SupportsFloat],
        ],
    ) -> None: ...
    def log_metrics(self) -> None: ...
    def record_arrival(self) -> None: ...
    def record_completion(self) -> None: ...
    def record_decode_arrival(self) -> None: ...
    def record_decode_scheduled(self) -> None: ...
    def record_first_scheduled(self) -> None: ...
    def record_first_token(self) -> None: ...
    def record_token(self) -> None: ...
    @property
    def arrival_time(self) -> float | None: ...
    @arrival_time.setter
    def arrival_time(self, arg0: typing.SupportsFloat | None) -> None: ...
    @property
    def avg_itl(self) -> float | None: ...
    @property
    def avg_itl_exclude_first(self) -> float | None: ...
    @property
    def avg_tpot_with_queueing(self) -> float | None: ...
    @property
    def avg_tpot_wo_queueing(self) -> float | None: ...
    @property
    def completion_time(self) -> float | None: ...
    @completion_time.setter
    def completion_time(self, arg0: typing.SupportsFloat | None) -> None: ...
    @property
    def decode_arrival_time(self) -> float | None: ...
    @decode_arrival_time.setter
    def decode_arrival_time(self, arg0: typing.SupportsFloat | None) -> None: ...
    @property
    def decode_queue_time_ms(self) -> float | None: ...
    @property
    def decode_scheduled_time(self) -> float | None: ...
    @decode_scheduled_time.setter
    def decode_scheduled_time(self, arg0: typing.SupportsFloat | None) -> None: ...
    @property
    def e2e_latency(self) -> float | None: ...
    @property
    def first_scheduled_time(self) -> float | None: ...
    @first_scheduled_time.setter
    def first_scheduled_time(self, arg0: typing.SupportsFloat | None) -> None: ...
    @property
    def first_token_time(self) -> float | None: ...
    @first_token_time.setter
    def first_token_time(self, arg0: typing.SupportsFloat | None) -> None: ...
    @property
    def itl_samples(self) -> list[float]: ...
    @itl_samples.setter
    def itl_samples(
        self, arg0: collections.abc.Sequence[typing.SupportsFloat]
    ) -> None: ...
    @property
    def last_token_time(self) -> float | None: ...
    @last_token_time.setter
    def last_token_time(self, arg0: typing.SupportsFloat | None) -> None: ...
    @property
    def num_generated_tokens(self) -> int: ...
    @num_generated_tokens.setter
    def num_generated_tokens(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def num_prompt_tokens(self) -> int: ...
    @num_prompt_tokens.setter
    def num_prompt_tokens(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def p50_itl(self) -> float | None: ...
    @property
    def p99_itl(self) -> float | None: ...
    @property
    def queueing_time_ms(self) -> float | None: ...
    @property
    def seq_id(self) -> int: ...
    @seq_id.setter
    def seq_id(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def ttft(self) -> float | None: ...

class SequenceStatus:
    """
    Members:

      WAITING

      RUNNING

      FINISHED

      TO_BE_MIGRATED
    """

    FINISHED: typing.ClassVar[SequenceStatus]  # value = <SequenceStatus.FINISHED: 2>
    RUNNING: typing.ClassVar[SequenceStatus]  # value = <SequenceStatus.RUNNING: 1>
    TO_BE_MIGRATED: typing.ClassVar[
        SequenceStatus
    ]  # value = <SequenceStatus.TO_BE_MIGRATED: 3>
    WAITING: typing.ClassVar[SequenceStatus]  # value = <SequenceStatus.WAITING: 0>
    __members__: typing.ClassVar[
        dict[str, SequenceStatus]
    ]  # value = {'WAITING': <SequenceStatus.WAITING: 0>, 'RUNNING': <SequenceStatus.RUNNING: 1>, 'FINISHED': <SequenceStatus.FINISHED: 2>, 'TO_BE_MIGRATED': <SequenceStatus.TO_BE_MIGRATED: 3>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class ServerMetric:
    def __init__(self) -> None: ...
    def add_completed_request(self) -> None: ...
    def add_tokens(
        self, num_prompt: typing.SupportsInt = 0, num_generated: typing.SupportsInt = 0
    ) -> None: ...
    def get_metric_report(self, include_detailed: bool = False) -> str: ...
    def get_summary(self) -> dict: ...
    def record_decode_throughput(
        self, num_tokens: typing.SupportsInt, duration: typing.SupportsFloat
    ) -> None: ...
    def record_prefill_throughput(
        self, num_tokens: typing.SupportsInt, duration: typing.SupportsFloat
    ) -> None: ...
    def update_running_requests(self, count: typing.SupportsInt) -> None: ...
    def update_token_usage(
        self, dp_idx: typing.SupportsInt, num_tokens: typing.SupportsInt
    ) -> None: ...
    def update_waiting_migration_requests(self, count: typing.SupportsInt) -> None: ...
    def update_waiting_requests(self, count: typing.SupportsInt) -> None: ...
    @property
    def avg_decode_throughput(self) -> float | None: ...
    @property
    def avg_prefill_throughput(self) -> float | None: ...
    @property
    def current_decode_throughput(self) -> float | None: ...
    @property
    def current_prefill_throughput(self) -> float | None: ...
    @property
    def decode_throughput_samples(self) -> list[float]: ...
    @decode_throughput_samples.setter
    def decode_throughput_samples(
        self, arg0: collections.abc.Sequence[typing.SupportsFloat]
    ) -> None: ...
    @property
    def num_completed_requests(self) -> int: ...
    @num_completed_requests.setter
    def num_completed_requests(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def num_running_requests(self) -> int: ...
    @num_running_requests.setter
    def num_running_requests(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def num_waiting_migration_requests(self) -> int: ...
    @num_waiting_migration_requests.setter
    def num_waiting_migration_requests(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def num_waiting_requests(self) -> int: ...
    @num_waiting_requests.setter
    def num_waiting_requests(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def prefill_throughput_samples(self) -> list[float]: ...
    @prefill_throughput_samples.setter
    def prefill_throughput_samples(
        self, arg0: collections.abc.Sequence[typing.SupportsFloat]
    ) -> None: ...
    @property
    def start_time(self) -> float: ...
    @start_time.setter
    def start_time(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def token_usage_by_dp(self) -> dict[int, int]: ...
    @token_usage_by_dp.setter
    def token_usage_by_dp(
        self, arg0: collections.abc.Mapping[typing.SupportsInt, typing.SupportsInt]
    ) -> None: ...
    @property
    def total_generated_tokens(self) -> int: ...
    @total_generated_tokens.setter
    def total_generated_tokens(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def total_prompt_tokens(self) -> int: ...
    @total_prompt_tokens.setter
    def total_prompt_tokens(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def total_token_usage(self) -> int: ...
    @property
    def total_tokens(self) -> int: ...
    @total_tokens.setter
    def total_tokens(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def uptime(self) -> float: ...

def deserialize(
    data_ptr: typing.SupportsInt, data_len: typing.SupportsInt
) -> list[typing.Sequence]: ...
def postprocess_sequences(
    worker_states: ...,
    std: ...,
    dp_sp_seqs: collections.abc.Sequence[collections.abc.Sequence[typing.Sequence]],
    dp_sp_token_ids: collections.abc.Sequence[
        collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]]
    ],
    eos_id: typing.SupportsInt,
    is_prefill: bool,
    update_metrics: bool = True,
    thread_pool: ... = None,
) -> list[tuple[typing.Sequence, int]]: ...
def prepare_decode_cpp(
    dp_seqs: collections.abc.Sequence[typing.Sequence],
    sp_rank: typing.SupportsInt,
    sp_size: typing.SupportsInt,
    block_size: typing.SupportsInt,
    max_num_seqs: typing.SupportsInt,
) -> DecodeMetadata: ...
def prepare_prefill_cpp(
    seqs: collections.abc.Sequence[typing.Sequence],
    sp_rank: typing.SupportsInt,
    sp_size: typing.SupportsInt,
    block_size: typing.SupportsInt,
    max_num_seqs: typing.SupportsInt,
) -> PrefillMetadata: ...
def serialize(
    data_ptr: typing.SupportsInt,
    buffer_size: typing.SupportsInt,
    seqs: collections.abc.Sequence[typing.Sequence],
) -> int: ...
def update_seqs_inner_loop(
    dp_seqs: collections.abc.Sequence[typing.Sequence], sp_rank: typing.SupportsInt
) -> None: ...

ACTIVE: BlockContextSlot  # value = <BlockContextSlot.ACTIVE: 0>
FINISHED: SequenceStatus  # value = <SequenceStatus.FINISHED: 2>
LeastBatch: RoutingStrategy  # value = <RoutingStrategy.LeastBatch: 1>
LeastCache: RoutingStrategy  # value = <RoutingStrategy.LeastCache: 2>
MIGRATE: BlockContextSlot  # value = <BlockContextSlot.MIGRATE: 1>
RUNNING: SequenceStatus  # value = <SequenceStatus.RUNNING: 1>
RoundRobin: RoutingStrategy  # value = <RoutingStrategy.RoundRobin: 0>
SWAP: BlockContextSlot  # value = <BlockContextSlot.SWAP: 2>
TO_BE_MIGRATED: SequenceStatus  # value = <SequenceStatus.TO_BE_MIGRATED: 3>
WAITING: SequenceStatus  # value = <SequenceStatus.WAITING: 0>
