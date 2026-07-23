# LoongServe-style slowdown：短请求 CP 扩散与 GPU 负载偏斜

日期：2026-07-23 UTC

## 结论

这不是“负载不均衡”和“短请求被开 CP”二选一。两者同时存在，并由同一套
group-level `source_greedy` master 调度相互加强：

1. **更上游的结构性问题是短请求发生了无收益的 CP 扩散。** 本轮额外
   KV participant work 的约 98.3% 来自 prompt 不超过 1K、最终完整
   context 不超过 2,301 tokens 的请求；其中 99.91% 不是 admission 时
   必需的 CP，而是初始 DoP1 请求在后续 Decode master 切换后逐渐形成的
   CP2+。
2. **GPU master batch 不均衡是直接的同步执行瓶颈。** 忙时最忙 attention
   GPU 的 master batch 平均是 16 卡均值的 2.16 倍；同一个 Decode chunk
   必须等待最慢 rank。
3. **真正严重的慢轮通常是二者同时发生。** 在 `batch >= 800` 的轮次内，
   以各自中位数切分，低 CP/低偏斜的平均 ITL 是 90.57 ms，高 CP/低偏斜
   是 93.96 ms，低 CP/高偏斜是 94.31 ms，而高 CP/高偏斜达到
   118.07 ms。

因此，如果必须确定修复优先级，应先阻止短请求的 KV ownership 被 Decode
master churn 扩散，同时把 DP 内所有 group 的 master assignment 放进一个
全局负载目标。只修其中一个预计只能拿到部分收益。

## 数据与重放方法

分析对象：

- `ls_style_loop16_dp2sp8_r20_diag01_3.log`
- `ls_style_loop16_dp2sp8_r20_diag01_3.jsonl`
- 参考数据：
  `nanodeploy_original_issue001_2node_dp2sp8_r20_6min_rerun_20260717.log`

日志中的 `sp_size_hist_global` 不是估算值。调度器逐序列统计有 committed KV
的 rank 数，见 `scheduler.cpp:1365-1382`；对所有 remote KV owner 又会建立
Q send/result receive，见 `scheduler.cpp:1329-1362`。

为把 CP 归因到具体 prompt，本分析顺序重放了：

1. admission 的 `planned_kv_ranks`；
2. 每轮 `iteration_sequence_ids` 和 `iteration_master_assignments`；
3. KV consolidation 对 source rank 的删除。

重放覆盖 346 个 Decode outer step：

- 日志精确 sequence-rank work：401,107；
- 重放值：401,104；
- 339/346 个 step 完全匹配，累计只差 3，误差为 0.00075%。

所以 prompt 分桶的归因误差可以忽略。

## 短请求 CP 扩散

本轮共有 272,430 个 logical sequence-step。若所有请求始终 DoP1，
sequence-rank work 也应为 272,430；实际为 401,107，即多出 128,677。

重放得到的 128,674 份额如下：

| Prompt 长度 | 请求数 | sequence-step | CP2+ step 占比 | 平均 CP | 额外 rank-work | 额外 work 占比 |
|---|---:|---:|---:|---:|---:|---:|
| `<= 1K` | 6,892 | 267,914 | 39.24% | 1.472 | 126,502 | 98.31% |
| `4K–16K` | 230 | 2,205 | 36.78% | 1.457 | 1,007 | 0.78% |
| `64K–256K` | 20 | 569 | 24.60% | 1.246 | 140 | 0.11% |
| `> 256K` | 58 | 1,742 | 45.46% | 1.588 | 1,025 | 0.80% |

这批 workload 没有落在 `1K–4K` 和 `16K–64K` 的 prompt。

短请求的证据尤其明确：

- 6,892 个 `prompt <= 1K` 请求中，3,189 个（46.27%）曾经变成 CP2+；
- 这些请求的最终 context 平均 809，P99 1,288，最大 2,301，全部不超过
  4K；
- 它们产生 126,502 个额外 rank-work；
- 其中 126,386（99.91%）来自 admission 初始 DoP1、后续才扩散的请求。

因此这不是“为了装下长 prompt 而不得不开 CP”。绝大部分是短 context 在
Decode 过程中换 master 后，把不同的 16-token chunk 留在不同 KV rank 上。

`loop_count=16` 只减少 outer scheduler 调用频率，不会减少这个开销。一次
错误的 CP placement 会在 16 个 inner decode token 中重复触发 remote
attention；而且 master 迁移后，旧 rank 的 historical KV 不会自动删除。

## GPU 负载偏斜

从 `iteration_master_assignments` 重建每轮 16 张 attention GPU 的 master
batch。`batch >= 512` 的 249 个忙轮为：

| 指标 | 均值 | P50 | P90 |
|---|---:|---:|---:|
| total batch | 1,028.5 | 1,060 | 1,247 |
| 16 卡理想均值 | 64.3 | 66.3 | 77.9 |
| hottest GPU master batch | 138.3 | 133 | 187.4 |
| hottest / 16-card mean | 2.16x | 2.11x | 2.61x |
| sequence-weighted CP（逐轮均值） | 1.467 | 1.453 | 1.678 |
| ITL | 98.02 ms | 91.38 ms | 119.31 ms |

日志中的典型慢轮在
`ls_style_loop16_dp2sp8_r20_diag01_3.log:9166-9167`：

- total batch 1,198，16 卡理想均值 74.9；
- hottest master batch 194，即 2.59x；
- CP histogram 为 `{D1: 546, D2: 487, D3: 128, D4: 37}`；
- model runner 3,135.2 ms / 16，step ITL 199.23 ms；
- scheduler 本轮虽然也高达 52.43 ms，但主时间仍在 16-token
  model-runner/communication 路径。

参考 original 日志的 `batch >= 512` 轮，hottest/mean 只有约 1.11x。
在双方都有样本的 `batch=512–700` 区间：

- LS：平均 batch 601，hottest 81.2，hottest/mean 2.16x；
- original：平均 batch 580，hottest 40.2，hottest/mean 1.11x。

这说明 master imbalance 不是单纯由 LS 的总 batch 更大造成的。

不过这不是严格同机同版本 A/B，而且 LS 的 hottest KV utilization 接近
99%，original 参考轮约 26%，所以不能把跨运行的绝对 ITL 差值全部归给
master imbalance。

## 两个因素对 ITL 的相对解释力

在 LS 本轮内部，对 `batch >= 512` 的忙轮：

| 指标 | 与 ITL 的原始相关系数 | 控制 batch、max KV util 后 | 再控制另一个因素后 |
|---|---:|---:|---:|
| 平均 CP | 0.804 | 0.811 | **0.655**（再控制 master imbalance） |
| master hottest/mean | 0.671 | 0.684 | **0.335**（再控制平均 CP） |
| 短请求额外 rank-work | 0.698 | 0.798 | **0.679**（再控制 master imbalance） |

在更窄的 `batch >= 800` 区间：

- 平均 CP 在再控制 imbalance 后与 ITL 的 partial correlation 为 0.703；
- imbalance 在再控制平均 CP 后为 0.530。

这些是观测相关，不是严格因果系数，但有三点可用于判断：

1. 两个因素都有独立信号；
2. 短请求 CP/额外 rank-work 的信号更强、更稳定；
3. 两者正相关，且同时高时出现明显交互恶化。

`batch >= 800` 的 2×2 分桶如下：

| 平均 CP | master imbalance | 轮数 | 平均 ITL |
|---|---|---:|---:|
| 低 | 低 | 52 | 90.57 ms |
| 低 | 高 | 50 | 94.31 ms |
| 高 | 低 | 50 | 93.96 ms |
| 高 | 高 | 51 | 118.07 ms |

## 代码机制

### 1. Fast path 不以单请求 owner locality 或 prompt 长度为目标

`plan_iteration_masters_source_greedy()`：

- 按 group aggregate KV tokens/blocks 对 candidate ranks 排序，
  `sp_state_manager.cpp:983-1005`；
- 对当前请求顺序切连续 chunk，且
  `target_chunk=max(remaining/n_left, batch_per_master)`，
  `sp_state_manager.cpp:1051-1102`；
- owner-bucket placement 只是 fast path 失败后的 repair，
  `sp_state_manager.cpp:1107-1128`；
- 没有“短 context 保持 DoP1”或“切 master 前估算 CP 收益”的条件。

当前 core 实际使用
`ls_min_comp_bound_decoding_batch_size=128`，见 manifest 第 40 行；命令中的
`ls_decode_batch_per_master=64` 是另外一个没有进入该 planner 的配置项。

### 2. 每个 group 单独规划，combined 阶段不重新均衡

调度器逐 group 调用 source-greedy，
`scheduler.cpp:4578-4648`。随后 combined 阶段只是拼接已有 assignment、
累计 `combined_master_load` 并做可行性校验，
`scheduler.cpp:4654-4706`；它不会为 DP 的八张 SP rank 做第二次全局
rebalance。

多个 group 因而可以同时把较大的 chunk 放到同一物理 rank，形成 observed
hottest/mean 2.16x。

### 3. Master 切换会留下历史 KV，CP 只增不减

commit 新 master 时，代码明确保留 historical counts/tables，只把新的
pending token 和 chunk 放到 target rank，
`sp_state_manager.cpp:2204-2243`。所以一个原本 DoP1 的短请求只要经历过
不同 master，就会成为 CP2+。

本轮 531 个 admission batch 中有 513 个 `CAPACITY_APPEND`；226 个新 group
首次 observed allocation 大于 admission 自身 planned DoP。也就是说，
短请求虽按 DoP1 admission，却经 donor group merge 获得多 rank allocation，
随后 source-greedy 有机会把它们的 master 切到别的 rank。

本轮仅在尾部执行 7 次 consolidation、累计 stall 约 0.67 秒；忙时积累的
短 shard 基本没有被及时清理。

## 修复与判别实验建议

建议用三个同机同版本的 2-node A/B，将两个因素拆开：

1. **Sticky-local only**：allocation/group 不变；对最终 context 很短或
   committed KV 集中于单 rank 的请求，只要容量可行就保持当前 owner
   master。目标是把短请求平均 CP 从 1.47 压回接近 1。
2. **Global-balance only**：CP policy 不变；在 DP 内所有 group 规划完后，
   用全局 `combined_master_load` 对 assignment 做负载修正。目标是把
   hottest/mean 从 2.16x 压到接近 1.1x。
3. **Both**：同时启用 sticky-local 和全局均衡，验证上面观察到的交互收益。

三轮都应开启 worker CUDA-event forward/communication timing，至少记录：

- per-rank forward time 与同步 wait；
- Q/result SP 通信条数、字节和时间；
- prompt/final-context 分桶的 CP histogram；
- per-rank master batch、KV tokens/blocks；
- 同一 batch/KV-util 区间的 ITL。

预期 sticky-local 会首先降低 sequence-rank work 和 SP A2A；global balance
会首先降低 slowest-rank time。若两者同时开启后收益显著大于单项之和，就
能确认本轮日志显示的 CP×imbalance 交互。
