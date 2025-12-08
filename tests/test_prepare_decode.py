import unittest
from dataclasses import dataclass, field
from typing import Dict
from unittest.mock import MagicMock

import numpy as np
import torch

# ==============================================================================
# 1. 环境 Mock
# ==============================================================================


@dataclass
class HFConfig:
    num_key_value_heads: int = 4
    num_attention_heads: int = 32
    hidden_size: int = 4096
    vocab_size: int = 32000
    max_position_embeddings: int = 8192


@dataclass
class Config:
    max_num_seqs: int = 256
    max_num_send_seqs: int = 64
    max_num_recv_seqs: int = 64
    hf_config: HFConfig = field(default_factory=HFConfig)
    engine_id: str = "test_engine"


class MockBlockContext:
    def __init__(self, master_sp_idx):
        self.master_sp_idx = master_sp_idx


class MockSequence:
    def __init__(self, seq_id, master_sp_idx, kv_distribution: Dict[int, int]):
        self.seq_id = seq_id
        self.last_token = 101
        self.temperature = 1.0
        self._block_ctx = MockBlockContext(master_sp_idx)
        self._kv_distribution = kv_distribution
        self.len_seq = 100

    def block_ctx(self, engine_id):
        return self._block_ctx

    def context_len(self, engine_id, rank):
        return self._kv_distribution.get(rank, 0)

    def last_block_page_id(self, engine_id, rank):
        return 0

    def last_block_num_tokens(self, engine_id, rank):
        return 0

    def __len__(self):
        return self.len_seq

    def __repr__(self):
        return f"Seq(id={self.seq_id}, master={self._block_ctx.master_sp_idx}, kv={self._kv_distribution})"


class MockDistContext:
    def __init__(self, rank, size):
        self.attn_sp_rank = rank
        self.attn_sp_world_size = size


class MockCacheContext:
    def __init__(self):
        self.block_size = 16


# 全局变量控制 Mock 状态
CURRENT_RANK = 0
WORLD_SIZE = 4
CAPTURED_CONTEXT = {}


def get_dist_context():
    return MockDistContext(CURRENT_RANK, WORLD_SIZE)


def get_cache_context():
    return MockCacheContext()


def set_context(**kwargs):
    global CAPTURED_CONTEXT
    CAPTURED_CONTEXT = kwargs


# Mock CUDA
original_cuda = torch.Tensor.cuda


def mock_cuda(self, non_blocking=False):
    return self


torch.Tensor.cuda = mock_cuda

# Mock flash_mla
flash_mla = MagicMock()
flash_mla.get_mla_metadata.return_value = (None, None)

# ==============================================================================
# 2. ModelRunner (包含 Original 和 Optimized)
# ==============================================================================


class ModelRunnerTest:
    def __init__(self, config):
        self.config = config
        self.engine_id = config.engine_id
        self.enforce_eager = False
        self.graph_attn_compute_bs = [1, 2, 4]

    def prepare_block_tables(self, dp_seqs):
        return torch.zeros((len(dp_seqs), 10), dtype=torch.int32)

    # --- Original Implementation (User's Logic) ---
    def prepare_decode_original(self, dp_seqs, is_dummy=False):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        sp_rank = get_dist_context().attn_sp_rank
        sp_size = get_dist_context().attn_sp_world_size

        for seq in dp_seqs:
            if seq.block_ctx(self.engine_id).master_sp_idx == sp_rank:
                input_ids.append(seq.last_token)
                positions.append(len(seq) - 1)
                slot_mapping.append(
                    seq.last_block_page_id(self.engine_id, sp_rank)
                    * get_cache_context().block_size
                    + seq.last_block_num_tokens(self.engine_id, sp_rank)
                    - 1
                )

        sp_seqs = [
            [
                seq
                for seq in dp_seqs
                if seq.block_ctx(self.engine_id).master_sp_idx == sp_idx
            ]
            for sp_idx in range(sp_size)
        ]
        sp_num_seqs = [len(seqs) for seqs in sp_seqs]

        context_lens_for_attn = []
        max_num_seqs = self.config.max_num_seqs

        for sp_idx in range(sp_size):
            current_sp_num = sp_num_seqs[sp_idx]
            current_context_lens = [0] * max_num_seqs
            for seq_id in range(current_sp_num):
                ctx_len = sp_seqs[sp_idx][seq_id].context_len(self.engine_id, sp_rank)
                current_context_lens[seq_id] = ctx_len
                if ctx_len > 0:
                    context_lens_for_attn.append(ctx_len)
            context_lens.append(current_context_lens)

        send_req_num = 0
        recv_req_num = 0

        for sp_idx in range(sp_size):
            if sp_idx == sp_rank:
                continue
            for seq_id in range(sp_num_seqs[sp_idx]):
                if context_lens[sp_idx][seq_id] > 0:
                    recv_req_num += 1

        for seq in sp_seqs[sp_rank]:
            has_remote_kv = False
            for remote_rank in range(sp_size):
                if remote_rank == sp_rank:
                    continue
                if seq.context_len(self.engine_id, remote_rank) > 0:
                    has_remote_kv = True
                    break
            if has_remote_kv:
                send_req_num += 1

        if (
            send_req_num > self.config.max_num_send_seqs
            or recv_req_num > self.config.max_num_recv_seqs
        ):
            raise ValueError("Exceeds limits")

        global_context_lens = [
            [
                (
                    sp_seqs[sp_rank][seq_id].context_len(self.engine_id, sp_idx)
                    if seq_id < sp_num_seqs[sp_rank]
                    else 0
                )
                for seq_id in range(self.config.max_num_seqs)
            ]
            for sp_idx in range(sp_size)
        ]

        sp_valid_request_counts = [
            sum(1 for val in sp_ctx_len if val > 0) for sp_ctx_len in context_lens
        ]

        q_slice_get = []
        for seq_id in range(sp_num_seqs[sp_rank]):
            if context_lens[sp_rank][seq_id] > 0:
                q_slice_get.append(seq_id)

        q_slice_fill = []
        current_pos = 0
        for sp_idx in range(sp_size):
            if sp_idx == sp_rank:
                for i in range(len(q_slice_get)):
                    q_slice_fill.append(current_pos + i)
                current_pos += sp_valid_request_counts[sp_idx]
            else:
                current_pos += sp_valid_request_counts[sp_idx]

        q_copy_mask = [1] * len(q_slice_get)

        res_slice_get_to_buffer_output = q_slice_fill.copy()
        res_slice_fill_to_buffer_output = [
            sp_rank * self.config.max_num_seqs + seq_index for seq_index in q_slice_get
        ]
        res_to_buffer_output_mask = [1] * len(res_slice_get_to_buffer_output)

        res_slice_get_to_buffer_input = []
        res_slice_fill_to_buffer_input = []

        current_attention_pos = 0
        for sp_idx in range(sp_size):
            if sp_idx == sp_rank:
                current_attention_pos += sp_valid_request_counts[sp_idx]
                continue

            for seq_id in range(sp_num_seqs[sp_idx]):
                if context_lens[sp_idx][seq_id] > 0:
                    res_slice_get_to_buffer_input.append(current_attention_pos)
                    res_slice_fill_to_buffer_input.append(
                        sp_idx * self.config.max_num_seqs + seq_id
                    )
                    current_attention_pos += 1

        res_to_buffer_input_mask = [1] * len(res_slice_get_to_buffer_input)

        def pad_to_size(arr, size, pad_value=-1):
            if len(arr) >= size:
                return arr[:size]
            else:
                return arr + [pad_value] * (size - len(arr))

        q_slice_get_tensor = torch.tensor(
            q_slice_get, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        q_slice_fill_tensor = torch.tensor(
            q_slice_fill, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        q_copy_mask_tensor = torch.tensor(
            q_copy_mask, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        res_slice_get_to_buffer_output_tensor = torch.tensor(
            res_slice_get_to_buffer_output, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        res_slice_fill_to_buffer_output_tensor = torch.tensor(
            res_slice_fill_to_buffer_output, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        res_to_buffer_output_mask_tensor = torch.tensor(
            res_to_buffer_output_mask, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        max_num_send_recv_seqs = max(
            self.config.max_num_send_seqs, self.config.max_num_recv_seqs
        )

        res_slice_get_to_buffer_input_tensor = torch.tensor(
            pad_to_size(res_slice_get_to_buffer_input, max_num_send_recv_seqs, -1),
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        res_slice_fill_to_buffer_input_tensor = torch.tensor(
            pad_to_size(res_slice_fill_to_buffer_input, max_num_send_recv_seqs, -1),
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        res_to_buffer_input_mask_tensor = torch.tensor(
            pad_to_size(res_to_buffer_input_mask, max_num_send_recv_seqs, 0),
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)

        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens_for_attn = torch.tensor(
            context_lens_for_attn, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        global_context_lens = torch.tensor(
            global_context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        block_tables = self.prepare_block_tables(dp_seqs)

        set_context(
            context_lens=context_lens,
            context_lens_for_attn=context_lens_for_attn,
            global_context_lens=global_context_lens,
            q_slice_get=q_slice_get_tensor,
            q_slice_fill=q_slice_fill_tensor,
            q_copy_mask=q_copy_mask_tensor,
            res_slice_get_to_buffer_output=res_slice_get_to_buffer_output_tensor,
            res_slice_fill_to_buffer_output=res_slice_fill_to_buffer_output_tensor,
            res_to_buffer_output_mask=res_to_buffer_output_mask_tensor,
            res_slice_get_to_buffer_input=res_slice_get_to_buffer_input_tensor,
            res_slice_fill_to_buffer_input=res_slice_fill_to_buffer_input_tensor,
            res_to_buffer_input_mask=res_to_buffer_input_mask_tensor,
        )

    # --- Optimized Implementation (NumPy & Vectorization) ---
    def prepare_decode_optimized(self, dp_seqs, is_dummy=False):
        sp_rank = get_dist_context().attn_sp_rank
        sp_size = get_dist_context().attn_sp_world_size
        max_num_seqs = self.config.max_num_seqs
        max_num_send_recv_seqs = max(
            self.config.max_num_send_seqs, self.config.max_num_recv_seqs
        )

        # 1. 预计算与数据提取
        all_master_sp_idx = np.array(
            [s.block_ctx(self.engine_id).master_sp_idx for s in dp_seqs], dtype=np.int32
        )

        # 2. 构建 Context Lens 矩阵
        context_lens_np = np.zeros((sp_size, max_num_seqs), dtype=np.int32)
        global_context_lens_np = np.zeros((sp_size, max_num_seqs), dtype=np.int32)
        sp_num_seqs = np.zeros(sp_size, dtype=np.int32)

        for i, seq in enumerate(dp_seqs):
            master_idx = all_master_sp_idx[i]
            local_seq_idx = sp_num_seqs[master_idx]

            if local_seq_idx < max_num_seqs:
                ctx_len = seq.context_len(self.engine_id, sp_rank)
                context_lens_np[master_idx, local_seq_idx] = ctx_len

                if master_idx == sp_rank:
                    for remote_rank in range(sp_size):
                        global_context_lens_np[remote_rank, local_seq_idx] = (
                            seq.context_len(self.engine_id, remote_rank)
                        )

            sp_num_seqs[master_idx] += 1

        # 3. 计算统计量
        sp_valid_request_counts = np.sum(context_lens_np > 0, axis=1)
        context_lens_for_attn_np = context_lens_np[context_lens_np > 0]

        recv_mask = context_lens_np.copy()
        recv_mask[sp_rank, :] = 0
        recv_req_num = np.sum(recv_mask > 0)

        # send_req_num 计算逻辑保持一致：只要当前Rank的序列在其他Rank有KV，就算一次Send
        # 注意：这里我们检查global_context_lens中当前Rank对应的那一片
        send_req_check = global_context_lens_np[:, : sp_num_seqs[sp_rank]]
        seq_needs_send_mask = np.any(
            np.delete(send_req_check, sp_rank, axis=0) > 0, axis=0
        )
        send_req_num = np.sum(seq_needs_send_mask)

        if (
            send_req_num > self.config.max_num_send_seqs
            or recv_req_num > self.config.max_num_recv_seqs
        ):
            raise ValueError(
                f"Exceeds limits: send={send_req_num}, recv={recv_req_num}"
            )

        # 4. 计算切片索引
        q_slice_get = np.where(context_lens_np[sp_rank, :] > 0)[0].astype(np.int32)

        offsets = np.zeros(sp_size + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(sp_valid_request_counts)

        start_pos = offsets[sp_rank]
        q_slice_fill = np.arange(
            start_pos, start_pos + len(q_slice_get), dtype=np.int32
        )
        q_copy_mask = np.ones(len(q_slice_get), dtype=np.int32)

        res_slice_get_to_buffer_output = q_slice_fill
        res_slice_fill_to_buffer_output = (sp_rank * max_num_seqs + q_slice_get).astype(
            np.int32
        )
        res_to_buffer_output_mask = np.ones(
            len(res_slice_get_to_buffer_output), dtype=np.int32
        )

        res_slice_get_to_buffer_input = []
        res_slice_fill_to_buffer_input = []

        for sp_idx in range(sp_size):
            if sp_idx == sp_rank:
                continue

            valid_indices = np.where(
                context_lens_np[sp_idx, : sp_num_seqs[sp_idx]] > 0
            )[0]

            if len(valid_indices) > 0:
                count = len(valid_indices)
                base_pos = offsets[sp_idx]

                slice_get = np.arange(base_pos, base_pos + count, dtype=np.int32)
                res_slice_get_to_buffer_input.append(slice_get)

                slice_fill = (sp_idx * max_num_seqs + valid_indices).astype(np.int32)
                res_slice_fill_to_buffer_input.append(slice_fill)

        if res_slice_get_to_buffer_input:
            res_slice_get_to_buffer_input = np.concatenate(
                res_slice_get_to_buffer_input
            )
            res_slice_fill_to_buffer_input = np.concatenate(
                res_slice_fill_to_buffer_input
            )
        else:
            res_slice_get_to_buffer_input = np.array([], dtype=np.int32)
            res_slice_fill_to_buffer_input = np.array([], dtype=np.int32)

        res_to_buffer_input_mask = np.ones(
            len(res_slice_get_to_buffer_input), dtype=np.int32
        )

        # 5. 张量传输
        def to_gpu_tensor(np_array, dtype=np.int32, pad_to=None, pad_val=0):
            if pad_to is not None and np_array.size < pad_to:
                padding = np.full(pad_to - np_array.size, pad_val, dtype=dtype)
                np_array = np.concatenate((np_array, padding))
            elif pad_to is not None and np_array.size >= pad_to:
                np_array = np_array[:pad_to]
            return (
                torch.from_numpy(np_array.astype(dtype))
                .pin_memory()
                .cuda(non_blocking=True)
            )

        context_lens = to_gpu_tensor(context_lens_np)
        context_lens_for_attn = to_gpu_tensor(context_lens_for_attn_np)
        global_context_lens = to_gpu_tensor(global_context_lens_np)

        q_slice_get_tensor = to_gpu_tensor(q_slice_get)
        q_slice_fill_tensor = to_gpu_tensor(q_slice_fill)
        q_copy_mask_tensor = to_gpu_tensor(q_copy_mask)

        res_slice_get_to_buffer_output_tensor = to_gpu_tensor(
            res_slice_get_to_buffer_output
        )
        res_slice_fill_to_buffer_output_tensor = to_gpu_tensor(
            res_slice_fill_to_buffer_output
        )
        res_to_buffer_output_mask_tensor = to_gpu_tensor(res_to_buffer_output_mask)

        res_slice_get_to_buffer_input_tensor = to_gpu_tensor(
            res_slice_get_to_buffer_input, pad_to=max_num_send_recv_seqs, pad_val=-1
        )
        res_slice_fill_to_buffer_input_tensor = to_gpu_tensor(
            res_slice_fill_to_buffer_input, pad_to=max_num_send_recv_seqs, pad_val=-1
        )
        res_to_buffer_input_mask_tensor = to_gpu_tensor(
            res_to_buffer_input_mask, pad_to=max_num_send_recv_seqs, pad_val=0
        )

        set_context(
            context_lens=context_lens,
            context_lens_for_attn=context_lens_for_attn,
            global_context_lens=global_context_lens,
            q_slice_get=q_slice_get_tensor,
            q_slice_fill=q_slice_fill_tensor,
            q_copy_mask=q_copy_mask_tensor,
            res_slice_get_to_buffer_output=res_slice_get_to_buffer_output_tensor,
            res_slice_fill_to_buffer_output=res_slice_fill_to_buffer_output_tensor,
            res_to_buffer_output_mask=res_to_buffer_output_mask_tensor,
            res_slice_get_to_buffer_input=res_slice_get_to_buffer_input_tensor,
            res_slice_fill_to_buffer_input=res_slice_fill_to_buffer_input_tensor,
            res_to_buffer_input_mask=res_to_buffer_input_mask_tensor,
        )


# ==============================================================================
# 3. 单元测试逻辑 (All Ranks)
# ==============================================================================


class VerifyOptimizationTest(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.runner = ModelRunnerTest(self.config)
        self.check_keys = [
            "context_lens",
            "context_lens_for_attn",
            "global_context_lens",
            "q_slice_get",
            "q_slice_fill",
            "q_copy_mask",
            "res_slice_get_to_buffer_output",
            "res_slice_fill_to_buffer_output",
            "res_to_buffer_output_mask",
            "res_slice_get_to_buffer_input",
            "res_slice_fill_to_buffer_input",
            "res_to_buffer_input_mask",
        ]

    def _run_all_ranks(self, seqs, case_name):
        global CURRENT_RANK, CAPTURED_CONTEXT
        print(f"\n=========================================")
        print(f"Test Case: {case_name}")
        print(f"=========================================")

        for rank in range(WORLD_SIZE):
            CURRENT_RANK = rank
            rank_prefix = f"[Rank {rank}]"

            # 1. Run Original
            CAPTURED_CONTEXT = {}
            self.runner.prepare_decode_original(seqs)
            ctx_original = CAPTURED_CONTEXT.copy()

            # 2. Run Optimized
            CAPTURED_CONTEXT = {}
            self.runner.prepare_decode_optimized(seqs)
            ctx_optimized = CAPTURED_CONTEXT.copy()

            # 3. Compare
            for key in self.check_keys:
                val_orig = ctx_original[key]
                val_opt = ctx_optimized[key]

                # Check 1: Dtype
                self.assertEqual(
                    val_orig.dtype,
                    val_opt.dtype,
                    f"{rank_prefix} Dtype mismatch for {key}",
                )
                # Check 2: Shape
                self.assertEqual(
                    val_orig.shape,
                    val_opt.shape,
                    f"{rank_prefix} Shape mismatch for {key}",
                )
                # Check 3: Values
                torch.testing.assert_close(
                    val_orig, val_opt, msg=f"{rank_prefix} Value mismatch for {key}"
                )

            print(f"{rank_prefix} PASS")

    def test_case_1_local_only(self):
        seqs = []
        seqs.append(MockSequence(0, 0, {0: 100}))
        seqs.append(MockSequence(1, 0, {0: 120}))
        seqs.append(MockSequence(2, 1, {1: 100}))
        seqs.append(MockSequence(3, 2, {2: 100}))
        seqs.append(MockSequence(4, 3, {3: 100}))
        self._run_all_ranks(seqs, "Local Only")

    def test_case_2_remote_read_recv(self):
        seqs = []
        # Master: 1, KV: 0 (100)
        seqs.append(MockSequence(10, 1, {0: 100}))
        self._run_all_ranks(seqs, "Remote Read (Recv)")

    def test_case_3_remote_write_send(self):
        seqs = []
        # Master: 0, KV: 1 (100)
        seqs.append(MockSequence(20, 0, {1: 100}))
        self._run_all_ranks(seqs, "Remote Write (Send)")

    def test_case_4_complex_mixed(self):
        seqs = []
        seqs.append(MockSequence(100, 0, {0: 100, 1: 100}))  # Split
        seqs.append(MockSequence(101, 0, {0: 200}))  # Local
        seqs.append(MockSequence(200, 1, {0: 100, 1: 100}))  # Remote Local
        seqs.append(MockSequence(300, 2, {2: 100}))  # Other
        self._run_all_ranks(seqs, "Complex Mixed")

    def test_case_5_hotspot_all_to_one(self):
        # All requests have KV on Rank 0
        seqs = []
        seqs.append(MockSequence(10, 1, {0: 100}))
        seqs.append(MockSequence(11, 2, {0: 100}))
        seqs.append(MockSequence(12, 3, {0: 100}))
        self._run_all_ranks(seqs, "Hotspot (All-to-One)")

    def test_case_6_broadcast_one_to_all(self):
        # One sequence split across ALL ranks
        seqs = []
        seqs.append(MockSequence(99, 0, {0: 2048, 1: 2048, 2: 2048, 3: 2048}))
        self._run_all_ranks(seqs, "Broadcast (One-to-All)")

    def test_case_7_ring_dependency(self):
        # 0->1, 1->2, 2->3, 3->0
        seqs = []
        seqs.append(MockSequence(100, 0, {1: 100}))
        seqs.append(MockSequence(101, 1, {2: 100}))
        seqs.append(MockSequence(102, 2, {3: 100}))
        seqs.append(MockSequence(103, 3, {0: 100}))
        self._run_all_ranks(seqs, "Ring Dependency")


if __name__ == "__main__":
    unittest.main()
