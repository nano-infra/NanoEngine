# Qwen3.5 WideEP 测试记录 (2026-03-06)

在实现了 WideEP (Attention TP + FFN EP) 及修复了 GatedDeltaNet 的 `out_proj` (`RowParallelLinear` -> `ReplicatedLinear`) 后，对 Qwen3.5 模型进行了 `non_disagg` 环境的测试。

测试结果显示，当前代码在 **`attn_tp=1`** 时可以正常工作，但在 **`attn_tp=2`** 时依然存在报错。

## 测试环境

- 脚本: `examples/non_disagg.py`
- 参数: `--kvcache_block_size 256 --max_num_seqs 32 --temperature 0 --prompt "Introduce yourself." --max_tokens 2048 --log_level INFO --enforce_eager false`

## 测试结果总结

| 模型                    | 并行配置                         | 数据类型    | 结果                                                                               |
| :---------------------- | :------------------------------- | :---------- | :--------------------------------------------------------------------------------- |
| `Qwen3.5-35B-A3B`       | `attn_dp=4, attn_tp=2, ffn_ep=8` | BF16 (默认) | ✅ **正确** (之前 GatedDeltaNet 修复后已可跑通)                                    |
| `Qwen3.5-397B-A17B`     | `attn_dp=8, attn_tp=1, ffn_ep=8` | BF16 (默认) | ✅ **正确**                                                                        |
| `Qwen3.5-397B-A17B`     | `attn_dp=4, attn_tp=2, ffn_ep=8` | BF16 (默认) | ❌ **报错** (分配 KV Cache 时捕捉 CUDA Graph `master_bs=1` 时失败，具体见上文日志) |
| `Qwen3.5-397B-A17B-FP8` | `attn_dp=4, attn_tp=2, ffn_ep=8` | FP8         | ❌ **报错**                                                                        |
| `Qwen3.5-397B-A17B-FP8` | `attn_dp=8, attn_tp=1, ffn_ep=8` | FP8         | ✅ **正确**                                                                        |

## 结论与下一步

- **Qwen3 (FP8 & FP16)**: `attn_tp=2` 的 WideEP 已完全跑通。
- **Qwen3.5 (包含 FP8 和 FP16)**: `attn_tp=1, ffn_ep=8` (纯 EP) 均能跑通，说明基础配置和转换层处理逻辑（在 TP=1 退化为 Identity）正常。
- **目前的堵点**: `Qwen3.5` (超大参数版 `397B-A17B`) 在开启 `attn_tp=2` 时，CUDA Graph 捕获或内存分配阶段仍存在问题。对于 `35B-A3B` 版本，修复了 GatedDeltaNet 后，`attn_tp=2` 是可以通过的（详见\[对话中的第一条结果\]）。这表明大模型版本在切分或计算上可能还存在与 GatedDeltaNet (或其他特化层) 相关、但未被完全兼容的张量形状问题，或者 KV Cache 分配大小的问题。
