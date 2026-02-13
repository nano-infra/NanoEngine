import uuid
from itertools import count
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from nanodeploy._cpp import SequenceMetric

from nanodeploy._cpp import (
    BlockContext as _CppBlockContext,
    SamplingParams as _CppSamplingParams,
    Sequence as _CppSequence,
    SequenceStatus as _CppSequenceStatus,
)

from nanodeploy.sampling_params import SamplingParams

Sequence = _CppSequence
BlockContext = _CppBlockContext
SequenceStatus = _CppSequenceStatus


def dump(self):
    from nanodeploy._cpp import BlockContextSlot

    res = f"Sequence(id={self.seq_id}, status={self.status}, len={self.num_tokens}, prompt_len={self.num_prompt_tokens})\n"
    res += f"  Tokens: {self.token_ids if len(self.token_ids) <= 20 else str(self.token_ids[:10]) + '...' + str(self.token_ids[-10:])}\n"
    for slot in [BlockContextSlot.ACTIVE, BlockContextSlot.MIGRATE]:
        try:
            ctx = self.block_ctx(slot)
            if ctx.engine_id:
                res += f"  Slot {slot.name}: engine={ctx.engine_id}, dp_idx={ctx.dp_idx}, master_sp={ctx.master_sp_idx}, sp_size={ctx.attention_sp}, dp_size={ctx.attention_dp}\n"
                res += f"    Blocks: {list(ctx.block_location)}\n"
                # Add sp_block_table info
                table_info = {}
                for sp_idx in range(ctx.attention_sp):
                    blocks = list(ctx.sp_block_table[sp_idx])
                    if blocks:
                        table_info[sp_idx] = blocks

                if table_info:
                    res += f"    BlockTable: {table_info}\n"

                # Add num_dispatched_tokens info
                dispatched_info = {
                    i: val for i, val in enumerate(ctx.num_dispatched_tokens) if val > 0
                }
                if dispatched_info:
                    res += f"    DispatchedTokens: {dispatched_info}\n"
        except Exception as e:
            pass
    return res


Sequence.dump = dump
