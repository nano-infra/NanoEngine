# 去中心化 DP2SP8：`ray.get` 后结果重建验证

## 结论

`LocalExecutor.run()` 中 `ray.get()` 返回后的 request-id
索引、校验和 token tuple 重建不是当前性能差距的根因。

- 原始路径总耗时为 `0.307 ms/engine-quantum`，不是此前估计的
  `30–40 ms/quantum`，原估计高了约 `98–130x`。
- `loop_count=16`，即使完全删除整段逻辑，ITL 的理论收益上限也只有
  `0.019 ms/token`。
- positional fastpath 将该段耗时降低了 `0.087 ms/quantum`
  (`-28.2%`)，对应理论 ITL 收益仅 `0.0054 ms/token`。
- fastpath 组的端到端 ITL/TPOT 没有改善，约 1% 的运行波动已经远大于
  这段代码可能产生的收益。

下一步不应继续优化这段 Python 重建。应在 `ray.get` 包围的约
`723–726 ms/quantum` 内继续拆分：

1. worker GPU/CUDA 完成；
2. `ModelRunner.run()` 返回；
3. Ray actor result ready；
4. driver `ray.get()` 返回。

注意：当前 `ray_get_latency_ms` 包含 worker 执行、返回结果的 Ray
传输与反序列化，不等同于纯 Ray 控制面开销。

## 测试配置

- 架构：去中心化 DP2SP8（2 节点、16 GPU）
- 到达率：40 req/s
- 发送时长：120 s
- 请求数：4800
- 数据集过滤：跳过总 token 数大于 910000 的请求
- SP backend：`hao_basic`
- `loop_count=16`
- `max_ingress_drain_ms=0`
- A：`NANODEPLOY_HIER_RESULT_FASTPATH=0`
- B：`NANODEPLOY_HIER_RESULT_FASTPATH=1`

A、B 两组均完成 4800/4800 个请求，无失败；输入 token
`22,269,119`、输出 token `2,872,121` 完全一致，逐请求
prompt/output token 规格无差异。

## `LocalExecutor` 直接打点

以下数据为两台 LocalEngine 的最终累计值；包含相同的 warmup，因此
用于同口径对比。

| 指标（ms/engine-quantum） | A 原始路径 | B positional fastpath | B - A |
| --- | ---: | ---: | ---: |
| `ray.get` 包围区间 | 722.768 | 726.298 | +3.530 |
| post-`ray.get` 重建总计 | 0.307 | 0.220 | -0.087 |
| 构建索引/顺序 token tuple | 0.210 | 0.185 | -0.025 |
| request-id 校验 | 0.044 | 0.004 | -0.039 |
| 按 expected 顺序打包 | 0.039 | 0.016 | -0.023 |
| 重建占 `ray.get` 包围区间 | 0.042% | 0.030% | -0.012 pp |

最终累计原始计数：

| 计数 | A 原始路径 | B positional fastpath |
| --- | ---: | ---: |
| engine-quantum samples | 520 | 518 |
| `ray_get_latency_ms_total` | 375839.116 | 376222.379 |
| `result_rebuild_latency_ms_total` | 159.519 | 114.043 |
| `result_index_latency_ms_total` | 109.318 | 96.005 |
| `result_validate_latency_ms_total` | 22.712 | 2.232 |
| `result_pack_latency_ms_total` | 20.352 | 8.488 |

## 端到端结果

| 指标 | A 原始路径 | B positional fastpath | B 相对 A |
| --- | ---: | ---: | ---: |
| TPOT w/queue Avg | 62.957 ms | 63.847 ms | +1.41% |
| TPOT w/queue P50 | 65.191 ms | 65.973 ms | +1.20% |
| TPOT w/queue P90 | 67.623 ms | 69.586 ms | +2.90% |
| TPOT w/queue P99 | 72.553 ms | 74.530 ms | +2.72% |
| hierarchical decode ITL Avg | 58.770 ms | 59.499 ms | +1.24% |
| benchmark runtime | 189.874 s | 190.004 s | +0.07% |
| output throughput | 15126.43 tok/s | 15116.13 tok/s | -0.07% |

B 的 ITL/TPOT 轻微退化不能归因于 fastpath：fastpath 直接计时只省下
`0.087 ms/quantum`，而 B 组的 `ray.get` 包围区间本身增加了
`3.530 ms/quantum`。逐请求路由也有 311/4800 个请求落到了不同的
LocalEngine，说明两次运行存在正常的调度与系统波动。

## 原始结果

- A summary:
  `bench_logs/result_rebuild_ab_a_r40_2min_20260728/A1_original/decentralized_dp2sp8/deepseek-v3/sharegpt4o-random_geminiissue_r0.01_n60000_60k/dp2sp8_seg64k_n4800_r40_bs192_LB_hier_rpLB_maxreq910000_hao_basic_result_rebuild_A1_original_r40_2min_decentralized_dp2sp8/20260728_172124.summary.json`
- A per-request:
  `bench_logs/result_rebuild_ab_a_r40_2min_20260728/A1_original/decentralized_dp2sp8/deepseek-v3/sharegpt4o-random_geminiissue_r0.01_n60000_60k/dp2sp8_seg64k_n4800_r40_bs192_LB_hier_rpLB_maxreq910000_hao_basic_result_rebuild_A1_original_r40_2min_decentralized_dp2sp8/20260728_172124.jsonl`
- B summary:
  `bench_logs/result_rebuild_ab_a_r40_2min_20260728/B_fast/decentralized_dp2sp8/deepseek-v3/sharegpt4o-random_geminiissue_r0.01_n60000_60k/dp2sp8_seg64k_n4800_r40_bs192_LB_hier_rpLB_maxreq910000_hao_basic_result_rebuild_B_fast_r40_2min_decentralized_dp2sp8/20260728_172659.summary.json`
- B per-request:
  `bench_logs/result_rebuild_ab_a_r40_2min_20260728/B_fast/decentralized_dp2sp8/deepseek-v3/sharegpt4o-random_geminiissue_r0.01_n60000_60k/dp2sp8_seg64k_n4800_r40_bs192_LB_hier_rpLB_maxreq910000_hao_basic_result_rebuild_B_fast_r40_2min_decentralized_dp2sp8/20260728_172659.jsonl`
