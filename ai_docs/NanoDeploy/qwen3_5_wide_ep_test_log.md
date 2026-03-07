# Qwen3.5 WideEP 测试记录 (2026-03-06)

在实现了 WideEP (Attention TP + FFN EP) 及修复了 GatedDeltaNet 的 `out_proj` (`RowParallelLinear` -> `ReplicatedLinear`) 后，对 Qwen3.5 模型进行了 `non_disagg` 环境的测试。

测试结果显示，在进一步修复了 `GenericBackendFactory` 的 `parallel_context` 传递缺失 bug 以及 `GatedDeltaNet` 等问题后，当前代码在 **`attn_tp=1`** 和 **`attn_tp=2`** 均已可以正常且正确地工作，彻底对齐了输出。

## 测试环境

- 脚本: `examples/non_disagg.py`
- 参数: `--kvcache_block_size 256 --max_num_seqs 32 --temperature 0 --prompt "Introduce yourself." --max_tokens 2048 --log_level INFO --enforce_eager false`

## 测试结果总结

| 模型                    | 并行配置                         | 数据类型    | 结果                                                                             |
| :---------------------- | :------------------------------- | :---------- | :------------------------------------------------------------------------------- |
| `Qwen3.5-35B-A3B`       | `attn_dp=4, attn_tp=2, ffn_ep=8` | BF16 (默认) | ✅ **正确** (之前 GatedDeltaNet 修复后已可跑通)                                  |
| `Qwen3.5-397B-A17B`     | `attn_dp=8, attn_tp=1, ffn_ep=8` | BF16 (默认) | ✅ **正确**                                                                      |
| `Qwen3.5-397B-A17B`     | `attn_dp=4, attn_tp=2, ffn_ep=8` | BF16 (默认) | ✅ **正确** (分配 KV Cache CUDA Graph 问题以及 TP=2 计算污染的 bug 均已完全修复) |
| `Qwen3.5-397B-A17B-FP8` | `attn_dp=4, attn_tp=2, ffn_ep=8` | FP8         | ✅ **正确**                                                                      |
| `Qwen3.5-397B-A17B-FP8` | `attn_dp=8, attn_tp=1, ffn_ep=8` | FP8         | ✅ **正确**                                                                      |

## 结论与下一步

- **Qwen3 (FP8 & FP16)**: `attn_tp=2` 的 WideEP 已完全跑通。
- **Qwen3.5 (包含 FP8 和 FP16)**: `attn_tp=1, ffn_ep=8` (纯 EP) 均能跑通，说明基础配置和转换层处理逻辑（在 TP=1 退化为 Identity）正常。
- **最新进展**: `Qwen3.5` 所有的 `attn_tp=2` 问题（包括 CUDA graph shape mismatch 以及底层 `parallel_context` 丢失导致的数值错误/污染）均已彻底排查并修复。在 `non_disagg` 模式下，各种尺度模型对应的 WideEP 规格配置已全面跑通并且输出完全正确。
