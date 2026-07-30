# 中心化/去中心化 DP2SP8：worker 完成边界验证

## 结论

本次同配置 120 秒对照没有复现此前 6 分钟测试中的 `+7.1%`
去中心化 TPOT 差距：

- 中心化 TPOT w/queue Avg 为 `61.894 ms`；
- 去中心化为 `62.241 ms`，只高 `0.347 ms`（`+0.56%`）；
- 去中心化总 drain 时间更短，输出吞吐反而高 `1.00%`。

新增的 worker 完成边界计时可以确认：

1. `worker 完成 -> ray.get 返回` 不是根因。中心化为
   `1.491 ms/quantum`，去中心化两路平均为
   `1.682 ms/engine-quantum`，只多 `0.190 ms/quantum`，折合
   `0.012 ms/token`。
2. 去中心化没有固定的 `30–40 ms/quantum` driver 控制开销。本轮
   worker 临界段反而比中心化低 `8.726 ms/quantum`。
3. 加上上一轮已测得的 post-`ray.get` 结果重建
   `0.307 ms/quantum`，Ray 返回和 Python 结果重建合计仍远不足以解释
   旧测试的数毫秒/token 差距。
4. 本轮 TPOT w/queue 的微小劣势来自首次调度前的 admission/queue
   等待，而不是 decode 执行变慢；去中心化在首 token 之后的同口径
   request-level decode proxy 反而快 `2.413 ms/token`。

因此，固定的 LocalExecutor/Ray 返回路径已经排除。旧 6 分钟差距是
**持续负载下才出现的状态相关问题**，不能用这次包含较长 drain 阶段的
120 秒聚合值直接定位。

旧日志给出了一个需要优先做因果验证的新线索：`least_batch` 只按
owner/request 数量路由，不按 prompt/KV/attention work 路由。旧的慢
运行虽然两路请求数和输出 token 几乎完全均衡，但估算 attention work
相差 `34.1%`；本次快运行只相差 `6.6%`。这目前是最值得验证的
状态相关假设，但尚不能仅凭跨轮相关性宣布为最终根因。

## 测试配置

- 2 节点、16 GPU
- DP2SP8、EP16、TP1
- SP backend：`hao_basic`
- 到达率：40 req/s
- 发送时长：120 s
- 请求数：4800
- 数据集过滤：跳过总 token 数大于 910000 的请求
- `loop_count=16`
- 去中心化 `max_ingress_drain_ms=0`
- 去中心化使用原始结果重建路径：
  `NANODEPLOY_HIER_RESULT_FASTPATH=0`

两组均完成 4800/4800，无 reject、无失败；输入 token
`22,269,119`、输出 token `2,872,121` 完全一致。

## 端到端结果

| 指标 | 中心 DP2 | 去中心 DP2 | 去中心相对中心 |
| --- | ---: | ---: | ---: |
| TPOT w/queue Avg | 61.894 ms | 62.241 ms | +0.56% |
| TPOT w/queue P50 | 64.816 ms | 64.368 ms | -0.69% |
| TPOT w/queue P90 | 67.214 ms | 66.769 ms | -0.66% |
| TPOT w/queue P99 | 69.279 ms | 72.321 ms | +4.39% |
| benchmark runtime | 190.142 s | 188.267 s | -0.99% |
| output throughput | 15,105.12 tok/s | 15,255.54 tok/s | +1.00% |
| TTFT Avg | 2.061 ms | 1,436.383 ms | +1,434.322 ms |

request-level TPOT 可以按相同字段精确拆成：

`E2E/output = (E2E-TTFT)/output + TTFT/output`

| request-level 均值 | 中心 DP2 | 去中心 DP2 | 去中心 - 中心 |
| --- | ---: | ---: | ---: |
| TPOT w/queue | 61.894 ms/token | 62.241 ms/token | +0.347 |
| 首 token 后 decode proxy | 61.890 ms/token | 59.477 ms/token | -2.413 |
| TTFT/output 贡献 | 0.004 ms/token | 2.764 ms/token | +2.760 |

即 `-2.413 + 2.760 = +0.347 ms/token`。本轮去中心化不是 decode
吞吐更差，而是约一个到两个 quantum 的首次 admission 等待抬高了
request-level TPOT，尤其影响短输出请求和 P99。

## worker 完成边界

worker 在 token tensor 完成 `.tolist()`、CUDA 结果已落到 CPU 后记录
wall-clock 时间；driver 在 `ray.get()` 返回后记录接收时间。中心化
driver 对 16 个 worker 取最大完成时间，去中心化每个 LocalEngine 对
本地 8 个 worker 取最大完成时间。

| 指标（ms/quantum） | 中心 DP2 | 去中心 E0 | 去中心 E1 | 去中心均值 | 均值差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| samples | 245 | 244 | 244 | 244 | -1 |
| actor submit | 1.412 | 3.235 | 3.155 | 3.195 | +1.784 |
| DLSlime `send_seqs` | 26.149 | 27.367 | 27.819 | 27.593 | +1.444 |
| worker observed critical | 759.473 | 750.748 | 750.745 | 750.747 | -8.726 |
| worker finish -> `ray.get` | 1.491 | 1.610 | 1.753 | 1.682 | +0.190 |
| executor begin -> `ray.get` | 760.965 | 752.360 | 752.499 | 752.430 | -8.535 |
| `ray.get` 包围区间 | 733.403 | 721.757 | 721.524 | 721.640 | -11.762 |
| worker finish skew | 0.126 | 0.075 | 0.082 | 0.079 | -0.047 |

注意这些区间有包含和重叠关系，不能逐行相加。尤其
`worker observed critical` 已包含 actor 提交、DLSlime 发送、worker
等待和 GPU 执行。即使把去中心化额外的 actor submit 与 `send_seqs`
完全按串行上界计算，也只有 `3.227 / 16 = 0.202 ms/token`；本轮实际
critical path 并未增加。

两路去中心化 worker critical 只差 `0.003 ms/quantum`，worker finish
skew 也只有约 `0.08 ms`，因此本轮没有出现一侧 LocalEngine 或单 rank
明显落后的证据。

## 为什么 120 秒结果不能解释旧的 6 分钟差距

| 运行 | 中心 TPOT w/q | 去中心 TPOT w/q | 去中心 decode ITL |
| --- | ---: | ---: | ---: |
| 旧 6 分钟 | 66.840 ms | 71.568 ms | 66.501 ms |
| 本次 2 分钟 | 61.894 ms | 62.241 ms | 58.012 ms |

旧去中心化运行的前 60 秒到达请求与本次前 60 秒表现接近；差距在持续
注入后扩大。本次在 120 秒停止注入，后续大量 token 在 batch 逐渐变小的
drain 阶段完成，最终边界均值也包含该 drain，因此没有进入与旧 6 分钟
运行相同长度的稳态过载区间。

旧 6 分钟去中心化运行的最终路由分布：

| 路由负载 | Engine 0 | Engine 1 | 较大/较小 |
| --- | ---: | ---: | ---: |
| 请求数 | 7,205 | 7,195 | 1.001x |
| 输出 token | 4,320,762 | 4,323,262 | 1.001x |
| prompt token | 37,319,564 | 29,063,155 | 1.284x |
| attention-work proxy | 19,866,212,629 | 14,816,633,034 | 1.341x |

本次 2 分钟去中心化运行：

| 路由负载 | Engine 0 | Engine 1 | 较大/较小 |
| --- | ---: | ---: | ---: |
| 请求数 | 2,420 | 2,380 | 1.017x |
| 输出 token | 1,439,031 | 1,433,090 | 1.004x |
| prompt token | 10,783,752 | 11,485,367 | 1.065x |
| attention-work proxy | 5,290,490,249 | 5,639,597,732 | 1.066x |

这里的 attention-work proxy 为每个请求
`output * prompt + output * (output - 1) / 2` 的累加，只用于比较两路
相对长上下文负载，不等同于精确 GPU 时间。

`RequestRouter._candidate_engine_ids()` 的 `least_batch` key 只有当前
owner count；prompt 长度和预估 block 数只在 `least_cache` 路径参与。
DP2SP8 又使用 EP16，某一 DP 侧较慢会通过跨 16 rank 的 MoE collective
进入全局关键路径。这能解释“请求数看起来均衡，但持续负载仍可能慢”，
但还需要同轮干预实验验证因果。

## 下一步的确认实验

要真正钉死旧 6 分钟差距，下一轮应保持 360 秒注入，并把边界统计改成
按 30 秒窗口输出，而不是只保留包含 drain 的全程均值。每个窗口同时
记录：

1. `worker observed critical`、`worker finish -> ray.get`、结果重建；
2. 每个 LocalEngine 的 active request 数、KV block 数、当前上下文
   token/attention-work；
3. 中心化每个 DP 的相同负载字段；
4. 请求实际 owner，确保两组可按同一请求规格对齐。

因果 A/B 使用同一 6 分钟请求流：

- A：当前 `router_policy=least_batch`；
- B：`router_policy=least_cache`，或离线按 attention-work 做确定性
  双路均衡。

如果 B 消除上下文负载偏斜并同步消除 worker critical/ITL 差距，就可
确认根因是 sticky router 的工作量估计；如果负载已均衡但 worker
critical 仍慢，再继续向 worker 内部拆 CUDA/collective。

## 原始结果

- 中心化 summary：
  `bench_logs/worker_boundary_r40_2min_20260728/central/centralized_dp2sp8/deepseek-v3/sharegpt4o-random_geminiissue_r0.01_n60000_60k/dp2sp8_seg64k_n4800_r40_bs192_LB_legacy_rpLB_maxreq910000_hao_basic_worker_boundary_central_r40_2min_centralized_dp2sp8/20260728_175119.summary.json`
- 去中心化 summary：
  `bench_logs/worker_boundary_r40_2min_20260728/decentral/decentralized_dp2sp8/deepseek-v3/sharegpt4o-random_geminiissue_r0.01_n60000_60k/dp2sp8_seg64k_n4800_r40_bs192_LB_hier_rpLB_maxreq910000_hao_basic_worker_boundary_decentral_r40_2min_decentralized_dp2sp8/20260728_175650.summary.json`
- 中心化 per-request：
  `bench_logs/worker_boundary_r40_2min_20260728/central/centralized_dp2sp8/deepseek-v3/sharegpt4o-random_geminiissue_r0.01_n60000_60k/dp2sp8_seg64k_n4800_r40_bs192_LB_legacy_rpLB_maxreq910000_hao_basic_worker_boundary_central_r40_2min_centralized_dp2sp8/20260728_175119.jsonl`
- 去中心化 per-request：
  `bench_logs/worker_boundary_r40_2min_20260728/decentral/decentralized_dp2sp8/deepseek-v3/sharegpt4o-random_geminiissue_r0.01_n60000_60k/dp2sp8_seg64k_n4800_r40_bs192_LB_hier_rpLB_maxreq910000_hao_basic_worker_boundary_decentral_r40_2min_decentralized_dp2sp8/20260728_175650.jsonl`
