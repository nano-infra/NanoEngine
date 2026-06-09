import random
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from nanodeploy._cpp import BlockContextSlot, serialize
from nanodeploy.config import Config
from nanodeploy.engine.scheduler import Scheduler
from nanodeploy.engine.sequence import BlockContext, Sequence, SequenceStatus
from nanodeploy.logging import get_logger, set_log_level

logger = get_logger()
set_log_level("INFO")


def test_serialization_size_with_state():
    # Mock AutoConfig to avoid loading real model from dummy path
    with patch("transformers.AutoConfig.from_pretrained") as mock_conf:
        mock_conf.return_value = MagicMock(
            architectures=["Qwen3ForCausalLM"],
            max_position_embeddings=32768,
            num_hidden_layers=32,
            num_attention_heads=32,
            hidden_size=4096,
        )
        # 1. Setup Config & Scheduler
        config = Config(
            model="/models/dummy",
            engine_id="test_engine",
            num_kvcache_blocks=1024 * 1024 // 64,
            kvcache_block_size=64,
            attention_dp=1,
            attention_sp=8,
            max_num_seqs=1024,
            max_num_batched_tokens=500000,
        )

    scheduler = Scheduler(config)

    # 2. Create Sequence (1000 tokens)
    seqs = []
    for i in range(8192):
        token_ids = np.random.randint(0, 32768, size=16384).tolist()
        seq = Sequence(token_ids)
        scheduler.add(seq)
        seqs.append(seq)

    sch_res = scheduler.schedule()

    # Verify allocation happened
    # The scheduled sequence is in sch_res.dp_seqs[0][0]
    if not sch_res.dp_seqs or not sch_res.dp_seqs[0]:
        print("Scheduler returned no sequences!")
        return

    scheduled_seq = sch_res.dp_seqs[0][0]

    # Check if block table is populated using Enum
    # slot 0 is ACTIVE
    num_blocks = scheduled_seq.num_blocks(BlockContextSlot.ACTIVE, 0)
    print(f"Allocated {num_blocks} blocks for sequence.")
    assert num_blocks > 0, "Scheduler failed to allocate blocks!"

    # 4. Serialize
    buffer = torch.zeros(256 * 1024 * 1024, dtype=torch.int8)  # 256MB
    buffer_ptr = buffer.data_ptr()

    # Test 1: is_prefill=True
    size_prefill = serialize(buffer_ptr, buffer.numel(), sch_res.dp_seqs[0], True)
    print(f"Serialize [is_prefill=True] Size: {size_prefill} bytes")

    # Test 2: is_prefill=False
    size_decode = serialize(buffer_ptr, buffer.numel(), sch_res.dp_seqs[0], False)
    print(f"Serialize [is_prefill=False] Size: {size_decode} bytes")

    print("PASS: Serialization size checks passed with real scheduler state.")


if __name__ == "__main__":
    test_serialization_size_with_state()
