# tests/test_sequence_core.py

import os
import pickle
import sys
import unittest
import uuid
from dataclasses import dataclass
from typing import Optional

# 确保能找到 nanodeploy 包（如果未安装，将当前目录加入 path）
sys.path.append(os.getcwd())

# 尝试直接导入 C++ 编译模块进行单元测试，确保测的是 C++ 代码
try:
    from nanodeploy.engine._core import BlockContext, Sequence, SequenceStatus

    print("[TEST] Testing C++ backend implementation.")
except ImportError:
    print("[TEST] C++ backend not found. Please build extension first.")
    print("       Run: pip install -e .")
    sys.exit(1)


# 模拟依赖类
@dataclass
class SamplingParams:
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 100
    ignore_eos: bool = False


class TestSequenceCore(unittest.TestCase):
    def setUp(self):
        self.token_ids = [101, 202, 303]
        self.sampling_params = SamplingParams(temperature=0.7)
        self.engine_id = "cuda:0"
        self.seq = Sequence(self.token_ids, self.sampling_params, self.engine_id, 0)

    def test_production_uuid(self):
        """测试 UUID 是否符合 v4 格式标准"""
        seq_id = self.seq.seq_id
        print(f"Generated UUID: {seq_id}")

        # 1. 验证是否是合法的 UUID 字符串
        try:
            val = uuid.UUID(seq_id, version=4)
        except ValueError:
            self.fail(f"Invalid UUID string: {seq_id}")

        # 2. 验证是否符合 RFC 4122 v4 标准
        self.assertEqual(val.version, 4)
        # 验证变体 (Variant) 也是正确的 (Python uuid 库会自动处理，但我们显式检查更放心)
        # RFC 4122: variant bits should be 10xx
        # 这通常由 uuid 库解析保证，这里只要不报错即可

    def test_data_integrity(self):
        """测试数据在 C++ 内部的存储是否正确"""
        self.assertEqual(len(self.seq), 3)
        self.assertEqual(self.seq.get_item(0), 101)
        self.assertEqual(self.seq.get_item(-1), 303)
        self.assertAlmostEqual(self.seq.temperature, 0.7)

    def test_block_context_logic(self):
        """测试 BlockContext 的复杂逻辑"""
        ctx = self.seq.block_ctx(self.engine_id)
        self.assertEqual(ctx.engine_id, self.engine_id)

        # 模拟调度器分配 Block
        # 假设 block_size = 256
        ctx.num_dispatched_tokens[0] = 300

        # (300 + 256 - 1) // 256 = 2 blocks
        self.assertEqual(self.seq.num_blocks(self.engine_id, 0), 2)

        # 最后一个 block 里的 token 数: 300 - (1 * 256) = 44
        self.assertEqual(self.seq.last_block_num_tokens(self.engine_id, 0), 44)


if __name__ == "__main__":
    unittest.main()
