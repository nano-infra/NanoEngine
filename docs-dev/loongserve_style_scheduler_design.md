# NanoDeploy DeepSeek-V3 Dummy Decode 场景下的 LoongServe-Style Multi-Master Core 调度设计

## 1. 文档范围

本文档只设计以下运行场景：

```python
mode="decode"
dummy_prefill=True
loop_count=1
```

目标模型和固定并行拓扑为：

```text
model family = DeepSeek-V3
attention_dp = 4
attention_sp = 8
attention_tp = 1
ffn_ep       = 32
ffn_dp       = 1
ffn_tp       = 1
```

请求携带已有 prompt tokens。NanoDeploy 为这些 tokens 建立逻辑 KV Cache placement，但不执行真实 Prefill model computation，也不从 Prefill engine 搬运真实 KV。

本文只实现 **LoongServe-style Decode multi-master core policy**：

- admission 时以 batch 为单位确定初始 KV DoP，并在该 batch 的初始 ranks 上均匀切分所有 prompt KV；
- 每个 scheduler-visible Decode iteration 重新计算 active masters；
- master ranks 按已使用 KV 从多到少选择，即 pack-full-first；
- 使用 LoongServe 开源实现的 threshold-sized greedy chunk 生成 master mini-batches；
- Decode compute-bound 或现有 ranks 的 append capacity 无法覆盖剩余 requests 时增加 masters；
- 新 master 只承接当前及后续 token 的 KV append，不迁移历史 KV；
- batch 变小时自然减少 active masters，仍持有历史 KV 的 rank 继续作为 passive participant。

本文明确不包含：

- 真实 Prefill、hybrid mode 或 P/D disaggregation；
- Prefill/Decode 之间的资源竞争；
- 基于真实 Prefill 通信结果的 proactive KV retention/migration；
- 跨 engine 或跨 DP 的 KV migration；
- Decode 历史 KV rebalance、KV evacuation 或 rank-releasing migration；
- ACTIVE/STAGING runtime KV relayout transaction；
- Ray actors、GPU、模型副本或 process group 的物理扩缩容；
- 对 LoongServe 自定义 Decode kernel 或通信/计算 overlap 的复现；
- 真实语言生成质量或 Dummy Prefill 数值语义验证。

LoongServe 参考：

- 论文：[LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism](https://arxiv.org/abs/2404.09526)
- 本地实现：`/mnt/nvme1n1/ml_research/linbinbin1/LoongServe`
- source-style planner 对齐 commit：`f6e8fc15f31ea6913b353b627545058db5eec321`

### 1.1 论文声明边界

本文回答的问题是：

> 在 Decode-only workload 和固定 `4 DP x 8 SP x EP32` 拓扑下，能否把 LoongServe 的 multi-master、compute/memory scale-up 和 no-history-migration 思想移植到 NanoDeploy，并通过每迭代重算 masters 改善 DeepSeek-V3 Decode 性能？

按本文实现的系统在论文中应称为：

```text
LS-Decode-Core (our NanoDeploy reimplementation)
```

或：

```text
LoongServe-style multi-master decode-only scheduler
```

不能称为完整的 `LoongServe`、`LoongServe reproduction` 或端到端 ESP baseline。它是共享 NanoDeploy model/data plane 的 scheduler-policy baseline，只用于 Decode 组件级对比。

### 1.2 NanoDeploy 中 master 的 DeepSeek-V3 MoE 语义

LoongServe 原系统中，一个 elastic instance 是一份完整的 TP 模型副本；增加 master 会增加 local layers 和 FFN 的计算资源。

NanoDeploy 的 DeepSeek-V3 执行语义不同：

- attention master rank 持有该 token 的 hidden state；
- master 执行该 token 的 attention-side master computation、MoE gate，以及 EP collective 的 source-side dispatch；
- expert tokens 进入固定 EP32 域，由对应 expert ranks 执行 expert computation；
- expert outputs 返回 token source rank，由 master 完成 combine 和后续 token-side computation；
- passive attention participants 只处理其持有的历史 KV 所对应的 partial attention，但仍需按 NanoDeploy 固定 EP32 collective ordering 参与执行。

单个 token 在一轮内只有一个 source/master；该 token 的 dispatch 和 combine 不拆到多个 source ranks。multi-master 只是在 batch 维度把不同 tokens 分配给不同 source ranks。

因此，本文中的 `master_dop` 表示：

```text
本轮承载 real Decode tokens，并作为 MoE dispatch/combine source 的 SP rank 数量
```

它不表示 EP world size，也不表示实际启用的 GPU 数量。EP32 始终不变。增加 masters 的潜在收益来自：

- 分散 master-side attention/QKV/output 工作；
- 分散 MoE gate、dispatch 和 combine 的 source-side 工作；
- 避免单一 source rank 承载过大的 Decode batch。

是否存在有效的 multi-master crossover 必须通过目标模型、硬件、dtype、CUDA Graph 和 EP32 backend 上的 profiling 证明，不能直接从 LoongServe 的 Llama/TP 配置外推。

### 1.3 固定通信拓扑

- 32 个 workers、模型权重、DP/SP/EP device mesh 和 DLSLIME full mesh 在启动时创建；
- 每个 DP 的 8 个 SP ranks 共置于同一物理机，形成独立的 attention allocation domain；
- running group 固定属于一个 DP，不跨 DP 使用 SP participant；
- 多个 Decode groups 可以共享一个 DP iteration，但其 committed rank allocations 不重叠；
- EP=32 的 collective ordering 对所有 workers 保持一致；
- Scheduler 继续为没有 real master work 的 SP rank 补 dummy sequence，`LLMEngine` 将每个 DP batch 展开到全部 SP/TP workers；
- 32 个 workers 在相同 `loop_count` 和 layer ordering 下进入固定 EP32 collective；LS-Decode-Core 只改变 real sequence-to-master assignment，不改变 collective cadence；
- 不创建动态 NCCL/process group。

## 2. Dummy Prefill 生命周期

当前 `dummy_prefill=True` 仍会产生一个 `is_prefill=true` 的控制 step，但它只用于 admission 和逻辑 KV 初始化：

1. `LLMEngine.add_request()` 将请求加入 `waiting_migration`。
2. `Scheduler.schedule()` 调用 `_schedule_prefill()` 选择 DP、初始 master 和逻辑 KV placement。
3. 该 step 返回 `is_prefill=true`。
4. `LLMEngine.step()` 不调用 `executor.migrate()`，也不运行 Prefill model。
5. Engine 追加一个 dummy token，完成 Decode 输入状态初始化。
6. 后续 step 才执行真实 Decode model。

本文使用以下术语：

- **admission step**：上述 Dummy Prefill 控制步骤；
- **decode iteration**：一次 scheduler-visible `_schedule_decode()` 与对应 executor run，并且只生成一个 token；
- **real decode batch**：本轮实际执行的非 dummy requests。

LS-Decode-Core 全部实现、测试和实验均硬性要求：

```text
loop_count = 1
```

因此一次 scheduler-visible Decode iteration 就是一次 token iteration，masters 每生成一个 token 重算一次，与 LoongServe 的策略粒度一致。`loop_count > 1` 不属于本文档支持范围，也不作为扩展 variant 验收。

## 3. Core 扩缩容语义

NanoDeploy workers 和通信域始终常驻。每个 Decode iteration 都从当前 committed KV placement 重新派生 master assignment：

- 不维护 sticky master policy；
- 不使用 scale-up/down hysteresis、stable steps 或 cooldown；
- active master 数随本轮 real batch 和 append capacity 直接变化；
- 新增 master 不搬历史 KV；
- 不再作为 master、但仍持有 KV 的 rank 保留为 passive participant；
- participant allocation 不通过 KV migration 主动收缩；某 rank 对该 group 的 committed KV 和 pending append 都归零时可以直接回收；group merge 只取 allocation 并集；
- 不执行历史 KV rebalance 或 evacuation。

这里需要区分：

```text
master role change          = 每迭代重新分配本轮 token 的 source/master
KV participant membership  = 由历史 KV placement 决定，通常跨迭代持续存在
```

master role change 是纯计划变化，但仍必须正确处理 scheduler 边界上的 pending input token，见第 8 节。

## 4. Master、KV Participant 与 Group

对每个 real sequence，本轮有且只有一个 master：

- 读取本轮 input token；
- 执行 master-side attention 和 MoE source-side gate/dispatch/combine；
- 为本轮 input token 写入新 KV；
- 接收各 KV participants 的 partial O/LSE；
- 生成下一个 sampled token。

任何持有该 sequence committed 历史 KV 的 rank 都是 participant。

定义：

```text
D_init(batch)     = admission 后该 batch 的 prompt KV 初始分布所使用的 SP rank 数量
master_dop(group) = 本轮至少负责一个 real sequence 的 master rank 数量
kv_dop(group)     = 本轮持有该 group 任意 real sequence committed KV 的 rank 数量
```

`D_init` 是 admission-time、batch-level 的初始 placement 属性，创建后不重算，只用于复现和解释后续状态。一个未 merge group 通常只含一个 initial batch；group merge 后保留多个 batch 的 immutable initial placement records，不把它们重写成一个新的 `D_init`。`master_dop` 每轮重算；在不做历史 KV migration 的前提下，`kv_dop` 通常只会随着新 masters 写入 KV 而增加，并在 sequences finish 或某 rank 对该 group 的 KV 归零后下降。第一次 Decode scale-up 或 group merge 后三者不要求相等。

每个 Decode group 维护：

```text
DecodeGroupState
  group_id
  dp_idx
  sequences
  initial_batch_placements       # [{batch_id, sequences, D_init, initial_kv_ranks}, ...]
  allocated_attention_ranks
  kv_participants              # 从 sequence placement 派生
  last_iteration_masters       # 仅用于日志，不是下一轮约束
```

sequence 到本轮 master 的权威数据面字段仍使用 `BlockContext.master_sp_idx_`，但它是 **iteration-scoped assignment**，不能被调度策略当作 sticky preference。

### 4.1 Group 生命周期

- 一个 admission step 内接纳到同一 DP 的 compatible requests 默认形成一个 group；
- compatible 表示模型、dtype、并行拓扑、Decode backend 和执行 cadence 完全相同；本文所有 groups 的 `loop_count` 固定为 1；
- group 在 sequences 全部 finish/preempt 前持续存在；
- 同一 DP 内各 group 的 `allocated_attention_ranks` 在 committed 状态下两两不交；
- group 的 masters 和 KV participants 必须属于其 allocation；
- allocation 可以在 no-history-migration scale-up 时加入全局未分配 rank；
- allocation 不通过搬运 KV 缩小；zero-KV、zero-pending 的 rank 可以直接回收。

当没有未分配 rank，而某 group 无法满足本轮 compute/memory master demand 时，allocator 按以下顺序处理：

1. 尝试使用该 group allocation 内已有的 passive participants 作为 masters；
2. 若仍不足，与同 DP 中最老的 compatible group merge，取两个 allocations 的并集；
3. merge 后仍无法形成合法 append assignment 时，沿用现有 preemption；
4. 新 group admission 时若没有可分配 rank，则直接并入同 DP 最老 compatible group，或延迟 admission。

group merge 不搬 KV，只合并 group membership 和 rank allocation。survivor 使用较小的 `group_id`；sequences 按原 admission order 稳定拼接；双方已有的 `initial_batch_placements` 原样保留，不重新生成任何 committed prompt placement。

## 5. Batch Size 与 Source-Style Scale Trigger

### 5.1 Real Decode Batch

对每个 group：

```text
B_group = 本轮 _schedule_decode() 实际选中的 real sequence 数量
```

必须排除：

- SP dummy sequences；
- skipped sequences；
- waiting/admission sequences；
- finished 或已 preempted sequences。

### 5.2 Compute Threshold

设经过独立 profiling 得到的 compute-bound tipping point 为 `T_compute`：

```text
T_compute = 单个 master 在当前 EP32 data plane 上保持高效的最大参考 Decode batch
```

`64` 只能作为开发期占位值：

```text
ls_decode_batch_per_master = 64
```

正式实验前必须对以下空间 profiling：

```text
batch_size x context_length x master_dop x kv_dop
```

至少覆盖 `master_dop = 1, 2, 4, 8`，并证明增加 masters 在目标 EP32 data plane 上确实改善 Decode latency。

实现不先计算权威的 target master count，而是直接使用 LoongServe 开源源码中的 threshold-sized greedy chunk：当剩余 requests 在剩余可用 ranks 上的平均数量严格大于 `T_compute` 时加入新 rank；每个 master 以 `T_compute` 作为目标 chunk 下界，同时受 remaining requests 和 append capacity 限制。最终 `master_dop` 是 greedy plan 的输出。

### 5.3 Append Capacity

每个 master 必须能容纳本轮 input token 的 KV append。容量判断必须逐请求进行，至少考虑：

- pending input token 的目标 master；
- 当前 committed KV placement；
- destination tail block 的剩余空间；
- 固定的一次 token append；
- 每 rank free blocks；
- `reserved_blocks_per_req`；
- `max_num_seqs`。

本文不涉及 prefix cache 命中、prefix hash 或缓存复用。这里的 capacity 只表示 dummy Decode 仍会真实消耗的 KV block/page 与 metadata capacity。

对候选 rank 和 stable-order request suffix，定义只读模拟：

```text
estimate_append_capacity(rank, requests):
  free_blocks = rank 当前 free blocks
  accepted = 0
  for request in requests:
    if request 在该 rank 上有可用 tail slot:
      不消耗新 block
    elif free_blocks > 0:
      free_blocks -= 1
    else:
      break

    if 继续接纳会超过 max_num_seqs 或 CUDA Graph/metadata capacity:
      break
    accepted += 1
  return accepted
```

由于每条 real request 每轮只 append 一个 token，该模拟不需要 prefix-cache 逻辑，也不要求求解全局最优 matching。若 source-style greedy plan 无法覆盖全部 requests，则加入一个未分配 rank 后从头重新 plan；没有 rank 可加时再 merge compatible group 或沿用现有 preemption。绝不通过搬历史 KV 来腾空间，也不声称求得最少 memory master 数。

## 6. 每迭代 Master Planning

每个 Decode iteration 执行以下确定性算法。

### 6.1 候选 rank

候选集合首先包含 group allocation 内的全部 ranks。额外 ranks 来自同 DP 内尚未分配的 ranks，并按以下稳定 key 排序：

```text
(-estimated_append_capacity, sp_idx)
```

当 compute threshold 或 append capacity 要求更多 masters 时，一次加入一个额外 rank，扩大 allocation，并从头重新生成完整 plan。

候选 rank 只在满足以下条件时可成为 master：

- 能为至少一个 pending input token 提供合法 append slot；
- 本轮 master sequence 数不超过 `max_num_seqs`；
- 不违反 CUDA Graph 的 master/attention batch 容量；
- 属于该 group 最终 allocation。

### 6.2 Pack-Full-First 排序

候选 masters 按以下稳定 key 排序：

```text
(-group_used_kv_tokens,
 -group_used_kv_blocks,
  recv_count,
  sp_idx)
```

即优先选择已经持有较多本 group KV、但仍有 append capacity 的 ranks。这样能够：

- 避免无必要地扩散新 KV；
- 延缓 `kv_dop` 增长；
- 提高已有 participant 被直接复用为 master 的概率；
- 对齐 LoongServe 源码中的 pack-full-first 行为。

只有 compute threshold 或现有 allocation 的 greedy append capacity 要求扩容时，才加入 used KV 为 0 的新 rank。

### 6.3 Source-Style Threshold Chunk Assignment

group sequences 保持稳定 admission order。权威算法直接对齐 LoongServe 开源实现的 threshold-sized greedy chunk，而不是先计算 target master count 后做均衡切片：

```text
remaining = stable real sequence list
candidates = allocation 内 ranks，按 6.2 的 pack-full-first 顺序
extras = DP 内未分配 ranks，按 6.1 的顺序
plan = empty

while remaining 非空:
  跳过 append capacity 为 0 的 candidates；其中仍持有 KV 的 rank 保持 passive

  if 没有 append-capable candidate:
    if extras 非空:
      加入一个 extra，扩大 allocation，并从头重新 plan
    else:
      plan failure

  n_left = 当前及后续 append-capable candidate 数量

  while floor(len(remaining) / n_left) > T_compute and extras 非空:
    加入一个 extra，扩大 allocation
    将该 extra 追加到 candidates
    n_left += 1

  rank = 下一个 append-capable candidate
  target_chunk = max(floor(len(remaining) / n_left), T_compute)
  capacity = estimate_append_capacity(rank, remaining)
  chunk_size = min(len(remaining), target_chunk, capacity)

  if chunk_size == 0:
    将 rank 标记为本轮 passive，继续
  else:
    将 remaining 的前 chunk_size 条 sequences 分配给 rank
    从 remaining 删除该 contiguous prefix
```

例如在容量充足时：

```text
B=160, T_compute=128 -> master slices [128, 32]
B=300, T_compute=128 -> master slices [128, 128, 44]
B=64,  T_compute=128 -> master slices [64]
```

如果全部已有 candidates 用尽后仍存在 remaining sequences，则加入一个同 DP 的未分配 rank 并从头重新 plan；无 rank 可加时尝试 group merge，仍失败才 preempt/delay。任一失败路径都不进行部分 commit。

同一个 scheduler state 必须得到相同的 master rank 顺序和 sequence slices。

### 6.4 Plan/Validate/Commit

```text
plan
  -> 生成本轮 candidate masters、sequence slices 和 pending append slots

validate
  -> 检查所有 request capacity、group allocation、collective shape 和不变量

commit
  -> 一次性更新 iteration master assignment 和 pending token destination
```

commit 不修改任何 committed 历史 KV ownership，也不清空历史 block tables。

## 7. Scale-Up 与自然 Scale-Down

### 7.1 Compute/Memory Scale-Up

当 source-style planning 因以下任一原因需要更多 masters 时触发 scale-up：

- `floor(remaining / n_left) > T_compute`，即 compute-triggered；
- 现有 candidates 的 greedy append capacity 无法覆盖 remaining sequences，即 memory-triggered。

处理顺序为：

1. 优先启用 allocation 内已有 passive KV participants；
2. 再加入 DP 内未分配 ranks；
3. 为新增 masters 安装本 group 的 request metadata；
4. 将部分 sequences 的 pending input token 指派到新增 masters；
5. 从本轮开始在新 masters 上写入新 KV；
6. 历史 KV 完全保留在原 ranks。

Core baseline 不存在 scale-up rebalance path：

```text
compute scale-up historical KV bytes copied = 0
memory scale-up historical KV bytes copied  = 0
```

### 7.2 自然 Master Scale-Down

当 real batch 下降时，下一 iteration 直接使用更少 masters：

- 不等待 stable steps；
- 不使用 cooldown；
- 不执行 master-only scale-down transaction；
- 不搬历史 KV；
- 不再作为 master 的 rank 若持有 KV，继续作为 passive participant；
- 某 rank 对该 group 已无 committed KV、无 pending append 且本轮不是 master 时，可以直接从 allocation 回收；
- 仍持有 KV 的 rank 不释放。

因此 Core 中没有独立的 scale-down state machine。每轮 master planning 的输出就是本轮有效 `master_dop`。

## 8. Pending Input Token 与 Master 切换

NanoDeploy scheduler 边界上存在一个必须显式建模的状态：刚刚 sampled 的 token 已进入 sequence token list，但其 KV 要到下一次 forward 才进入 committed history。

因此必须区分：

```text
committed KV history visible to attention
pending input token not yet appended by the next Decode forward
```

实现可以增加显式字段，也可以使用等价状态，但语义必须满足：

```text
PerSequenceDecodeFrontier
  committed_kv_tokens_per_sp
  pending_token_present
  pending_token_target_sp
```

本轮重新选择 master 时：

1. committed 历史 KV 的 placement 完全不变；
2. pending input token 的 destination 改为本轮 target master；
3. 在 target master 预留正确的 append slot；
4. 不复制 pending token KV，因为它尚未进入 committed history；
5. forward 将 pending token 的 KV 写入 target slot；
6. attention local context 必须包含刚写入的当前 token；
7. forward 后该 token 进入 committed history，新 sampled token 成为下一轮 pending input。

Dummy Prefill 建立的 prompt slots 也按 committed logical history 参与 shape、block table 和 attention planning；这些 slots 的数值内容不作语言或 logits 正确性声明。

不得仅修改 `master_sp_idx_` 而把 pending token 的计数、block reservation 和 slot mapping 留在旧 master。

若第一版不引入显式 frontier，必须提供一个等价的 `reassign_pending_append()` 原子操作，并为以下场景增加测试：

- new master 在该 sequence 上已有历史 KV；
- new master 对该 sequence 的历史 KV 为 0；
- pending token 触发新物理 block；
- old master 的 tail block 只为 pending token 预留；
- `loop_count=1` 连续每轮切换 master。

## 9. 固定 DLSLIME Mesh 上的数据面

本 feature 复用 NanoDeploy 现有 DLSLIME Decode data plane，不复刻 LoongServe 的自定义 NCCL overlapped kernel。

本场景的 DLSLIME 运行配置是硬性合同：

```python
use_dlslime_rpc = True
sp_backend = "hao_basic"
fixed_sp_size = 0
```

其中两项 DLSLIME 用途必须区分：

- `use_dlslime_rpc=True` 要求 scheduler/engine 到 workers 的 sequence 传输使用现有 DLSLIME RPC path，不切换为直接传递 sequence 的非 DLSLIME 分支；
- `sp_backend="hao_basic"` 要求 Decode attention 的 Q/O/LSE 通信使用 `hao_basic` backend，并使用支持 Q offsets 的 native DLSLIME path；不允许静默回退到 compat、`legacy_ll`、`nccl` 或 `nccl_compact` backend；native symbols 或 Q-offset 能力不可用时应直接报错。

两者服务于不同的数据通路，但在 LS-Decode-Core 中都必须启用。所有正确性、性能和 baseline 对比必须保持这三项配置完全相同。

调度器的权威输出是：

```text
group membership
iteration sequence-to-master assignment
committed logical KV placement
pending append destination
```

由这些状态统一派生：

- `q_mask`；
- `res_lse_mask`；
- `q_offsets` 和 Q/Res/LSE packing metadata；
- local/global context lengths；
- `context_lens_for_attn`；
- block tables 和 slot mapping；
- `attention_compute_bs` 与 `sp_comm_bs`。

对任意 sequence 必须满足：

```text
Q targets == ranks owning committed logical KV, excluding self edge
O/LSE target == the iteration master
local attention on master includes the current appended token
block tables and context lengths describe the same placement
```

当前 `use_sp_a2a` 不能只以 `active KV ranks > 1` 判断。正确条件至少为：

```text
use_sp_a2a = exists(sequence, owner_sp != iteration_master_sp)
```

否则，当 sequence 的全部历史 KV 只在一个旧 rank，而本轮 master 是一个 zero-history 新 rank 时，会错误地走 local-only attention。

该条件在整个 DP iteration 的 real sequences 上求值；同一 DP 的 8 个 SP ranks 必须得到相同的 `use_sp_a2a`，并以相同顺序进入对应 collective，不能由各 rank 或各 group 独立决定是否调用。

本文不把 DLSLIME 内部是否实现 communication/computation overlap 作为 LS-Decode-Core 验收项，但所有对比方法必须共享完全相同的 DLSLIME backend、CUDA Graph mode 和 data-plane 配置。论文中不得把底层 overlap 性能归因于该 scheduler policy。

DLSLIME full mesh 表示 8 个 SP ranks 之间的通信连接在启动时固定建立，不表示每个请求的 KV 都分布到全部 8 个 ranks。`fixed_sp_size=8` 会强制每个请求使用 8 个 KV ranks，语义与本设计冲突，因此必须保持 `fixed_sp_size=0`。

## 10. Admission 与初始 KV Placement

Admission step 只负责：

- 接纳 requests；
- 选择 DP；
- 形成或合并 Decode group；
- 确定 batch-level `D_init` 和有序 `initial_kv_ranks`；
- 在该 batch 的 `D_init` 个 ranks 上为所有 prompt KV 生成 uniform logical placement；
- 为 admission 后的 pending dummy token 设置 provisional target；
- 分配逻辑 blocks；
- 执行已有 dummy token append。

Admission 不运行 BS threshold scale policy，也不执行任何 KV migration。

### 10.1 `D_init` 选择

`D_init(batch)` 是 batch 级属性，表示 admission 后该 batch 的 prompt KV 初始分布所使用的 SP rank 数。它不是首轮或任意一轮的 `master_dop`。若后续发生 group merge，每个 admission batch 继续保留自己的 `D_init` 和 `initial_kv_ranks`。

配置语义为：

```text
ls_decode_initial_kv_dop == 0
  -> 自动选择能容纳整个 admission batch prompt KV 和 reservation 的最小 DoP

ls_decode_initial_kv_dop in [1, attention_sp]
  -> 强制使用指定 D_init；不可行时 admission 失败或延迟，不静默改值
```

自动模式按 `d = 1..attention_sp` 递增枚举。对每个 `d`：

1. 从该 DP 尚未分配的 ranks 中按 `(-free_blocks, sp_idx)` 选择前 `d` 个，得到有序 `R_init`；
2. 按第 10.2 节为本 admission batch 内全部 requests 生成 uniform prompt placement；
3. 逐 rank 计算所有 sequence prompt blocks 和 `reserved_blocks_per_req` headroom；
4. 校验 block capacity、receiver metadata、`max_num_seqs` 和 CUDA Graph/attention batch capacity；
5. 第一个可行的 `d` 即 `D_init`。

新 group 的初始 placement 只能使用未分配 ranks。若没有可行的 `D_init`，scheduler 可以先把该 admission batch 并入 compatible group，再从 survivor allocation 与同 DP 未分配 ranks 中为**新 batch**选择 `initial_kv_ranks` 并生成 uniform placement，或延迟 admission。merge 不得修改 survivor 中已有 batches 的 committed placement 或 `D_init` records；不允许两个未合并 groups 隐式共享 rank。

### 10.2 Batch-Level Uniform Prompt Placement

设 admission batch 内 sequence `s` 的 committed prompt KV token 数为 `L_s`，且：

```text
R_init = [r_0, r_1, ..., r_(D_init-1)]
base_s = floor(L_s / D_init)
rem_s  = L_s mod D_init
```

则：

```text
prompt_kv_tokens[s][r_i] = base_s + 1,  if i < rem_s
                            base_s,      otherwise

prompt_blocks[s][r_i] = ceil(prompt_kv_tokens[s][r_i] / block_size)
```

所有 requests 使用同一组 `R_init` 和同一个 `D_init`；调度粒度是 batch/group，而不是为每条 request 单独选择 SP DoP。逻辑上可视为 LoongServe 式 round-robin striped positions；在本文 dummy workload 中，权威状态只需要保持上述 token counts、block tables 和 context lengths 一致。

prompt tokens 在 admission 后属于 committed KV。Engine 追加的 dummy sampled token 是尚未产生 KV 的 pending input，不计入上述 uniform prompt split。Admission 只为它设置 provisional target；第一个真实 Decode iteration 可以通过第 8 节的 `reassign_pending_append()` 将它原子地改到 source-style planner 选择的 master。

### 10.3 生命周期与实验公平性

```text
admission batch:
  initial_kv_dop = D_init
  initial_kv_ranks = R_init
  若该 batch 独立形成新 group，则 group kv_dop 通常等于 D_init

decode scale-up:
  新 master 开始持有新 KV 后，kv_dop 可以大于 D_init

master scale-down:
  master_dop 可以下降；D_init 不变；仍持有 KV 时 kv_dop 不下降

finish / KV 归零:
  zero-KV、zero-pending rank 可以回收，kv_dop 可以自然下降
```

每个 admission batch 的 `D_init` 创建后不重算。运行时 group 的 participant 状态以其全部 sequences 当前 committed placement 派生的 `kv_dop` 为准。

为保证 scheduler-policy 实验可比较，所有方法必须使用完全相同的 initial KV placement。实验配置必须记录：

- `D_init` 和 `initial_kv_ranks`；
- placement strategy，Core 固定为 batch-level uniform；
- block size；
- per-sequence `num_dispatched_tokens`；
- provisional pending target；
- RNG seed。

Dummy Prefill 只用于 Decode 性能和 shape/kernel execution。本文不要求输出具有语言语义，也不要求与真实 Prefill logits 对齐。

## 11. 支持边界与配置

第一版只支持：

```text
mode == "decode"
dummy_prefill == true
scheduler_mode == "centralized"
DeepSeek-V3 model family
attention_dp == 4
attention_sp == 8
attention_tp == 1
ffn_ep == 32
ffn_dp == 1
ffn_tp == 1
loop_count == 1                 # 唯一支持值
sp_backend == "hao_basic"      # native Q-offset path
fixed_sp_size == 0
dynamic_sp_size_strategy == "legacy"
use_new_decode_dynamic_sp_scheduler == false
```

建议新增配置：

```python
enable_ls_decode_core_scheduler: bool = False
ls_decode_initial_kv_dop: int = 0       # 0=自动最小可行 D_init，1..8=强制值
ls_decode_batch_per_master: int = 64
ls_decode_enable_memory_scale_up: bool = True
```

不增加以下 Core 配置：

```text
scale_down_threshold
scale_down_stable_steps
scale_cooldown_steps
scale_up_kv_rebalance
rank_releasing_scale_down
max_migration_blocks_per_step
```

`Config.__post_init__()` 校验 `ls_decode_initial_kv_dop` 位于 `[0, attention_sp]`，并对其他不支持组合直接报错，不静默降级。

## 12. 代码接入点

### `nanodeploy/config.py`

- 增加四个 LS-Decode-Core 配置；
- 校验 DeepSeek-V3、并行拓扑、backend、`loop_count=1` 和互斥策略；
- feature flag 默认关闭。

### `csrc/nanodeploy/scheduler/scheduler.h/.cpp`

- 维护 `DecodeGroupState`；
- admission 时为新 batch 确定 `D_init`、`initial_kv_ranks` 和 batch-level uniform prompt placement；
- 在 `_schedule_decode()` 选出 real sequences 后、补 dummy sequences 前执行每轮 master planning；
- 确定性处理 group allocation/merge；
- `ScheduleResult` 记录 group、initial batch placement records、master DoP、KV DoP、source-style master slices 和 scale reason；
- admission、finish 和 preemption 同步 group state。

### `csrc/nanodeploy/scheduler/sp_state_manager.h/.cpp`

建议增加：

```text
select_initial_kv_dop(...)
plan_uniform_initial_kv_placement(...)
estimate_pending_append_capacity(...)
plan_iteration_masters_source_greedy(...)
validate_iteration_master_plan(...)
commit_iteration_master_plan(...)
reassign_pending_append(...)
get_group_used_kv_tokens(...)
get_active_master_count(...)
get_kv_participant_count(...)
```

不增加 runtime KV relayout、reserve/copy/commit/abort APIs。

### `csrc/nanodeploy/sequence/sequence.h/.cpp`

- 显式建模 committed KV frontier 与 pending input token，或提供等价状态；
- pending append destination 可以随 iteration master assignment 变化；
- committed 历史 `sp_block_table` 和 ownership 不随 master role change 而改变。

### `nanodeploy/worker/model_runner.py` 与 C++ Decode metadata

- 修正 zero-history new master 的 slot mapping；
- `use_sp_a2a` 基于 remote-owner relation，而不是仅基于 active-rank count；
- 从统一 snapshot 生成 masks、offsets、context lengths 和 block tables；
- 保持现有 DLSLIME data plane，不新增 LoongServe kernel。

### `nanodeploy/engine/llm_engine.py`

- 保留 Dummy Prefill admission 分支；
- 不调用 runtime KV migration；
- 记录 `D_init`、real batch、master DoP、KV DoP、source-style master slices 和 planning latency；
- dummy/admission 指标与 Decode 指标分离。

`nanodeploy/engine/ray_executor.py` 和 `nanodeploy/worker/cache.py` 不需要新增同 engine KV copy API。

## 13. 不变量

1. `D_init` 是 batch 级 admission 属性，创建后不重算；group merge 原样保留各 batch 的 initial placement record。
2. 每条 sequence 的 uniform prompt placement token 数之和等于其 committed prompt token 数，任意两个 initial ranks 的 token 数差不超过 1。
3. pending dummy token 不计入 uniform committed prompt placement。
4. 每个 running real sequence 在每轮有且只有一个 master。
5. iteration master 属于 sequence group allocation。
6. 所有 masters 都能容纳本轮 pending append，或 plan 整体失败。
7. committed 历史 KV ownership 在 master role change 时保持不变。
8. pending input token 在本轮 target master 上恰好 append 一次并进入 committed history。
9. remote KV owner 存在时必须走 SP attention，即使历史 KV 只在一个 rank。
10. 所有历史 KV holders 继续作为 Attention participants。
11. `sum(committed_kv_tokens_per_sp)` 等于 sequence committed KV token 数。
12. dummy sequences 不计入 real batch、master load 或 threshold。
13. master planning 每轮从当前状态重算，不以 last master 作为 sticky constraint。
14. candidate masters 使用确定性的 pack-full-first 顺序，sequence slices 使用 source-style threshold chunks。
15. compute/memory scale-up 不复制任何历史 KV bytes。
16. batch 下降时只减少 master roles；仅 zero-KV、zero-pending ranks 可直接回收，不迁移仍被引用的历史 KV。
17. 同一 DP 内不同未合并 groups 的 allocations 不重叠。
18. 所有 group 限制在单个 8-SP DP 域内。
19. feature 关闭时行为与当前 NanoDeploy 一致。

## 14. 指标与日志

至少记录：

- `real_decode_batch_size_per_group`；
- `initial_kv_dop_per_admission_batch`；
- `initial_kv_ranks_per_admission_batch`；
- `master_dop_per_group`；
- `kv_dop_per_group`；
- `master_batch_sizes_per_group`；
- `source_greedy_target_chunks_per_group`；
- `iteration_master_assignment`；
- `group_rank_allocation_per_dp`；
- `group_used_kv_tokens_per_rank`；
- `pending_append_blocks_per_master`；
- scale reason：`compute`、`memory`、`compute+memory`、`none`；
- new master ranks；
- reused passive-participant masters；
- historical KV migration bytes，Core 中必须恒为 0；
- scheduler planning latency；
- scale 前后 ITL；
- EP dispatch/combine latency 与 expert compute latency，如 profiler 可获得；
- KV capacity 导致的 preemption。

Admission 和 Decode 指标必须分开。论文中 `master_dop` 不得表述成使用 GPU 数或 EP DoP。

## 15. 测试方案

### 15.1 单元测试

- 自动 `D_init` 选择第一个能容纳整个 batch uniform prompt placement 和 reservation 的 DoP；
- 强制 `D_init=1..8` 时严格使用指定值，不可行时不静默改值；
- 每条 sequence 在 `D_init` ranks 上的 prompt token 数之和正确，任意两 rank 差不超过 1；
- pending dummy token 不计入 committed uniform prompt placement；
- 容量充足时，`B_group=1, T, T+1, 2T, 2T+1` 分别得到 `[1]`、`[T]`、`[T,1]`、`[T,T]`、`[T,T,1]` source-style slices；
- `B_group=0` 时无 active master、无空 chunk；
- dummy/skipped sequences 不计入 `B_group`；
- 每轮从当前 batch 重算 masters；
- batch 下降后立即减少 active masters，无 hysteresis；
- zero-KV、zero-pending participant rank 可以直接回收；
- used-KV 多的 rank 优先成为 master；
- existing passive participant 优先于 zero-history new rank；
- append capacity 无法覆盖 remaining sequences 时加入一个新 rank并从头重新 plan；
- plan 失败时不进行部分 assignment commit；
- group merge 使用确定 survivor 和 stable sequence order；
- compute/memory scale-up 的历史 placement 完全不变；
- pending input token 可以从旧 master 转到新 master；
- zero-history new master 的第一个 KV 写入 slot 正确；
- sole remote KV owner 时 `use_sp_a2a=True`；
- feature 关闭时保持原行为。

### 15.2 集成测试

- batch 从 32 增长到 160，再下降到 16；
- batch 在 threshold 两侧反复变化，masters 每轮按 source-style greedy chunks 直接变化；
- admission 后所有方法对每个 batch 使用完全相同的 `D_init`、`initial_kv_ranks` 和 uniform prompt placement；
- 长 prompt 造成 append pressure，验证 memory scale-up；
- 多 group 竞争 8 ranks，验证 allocation 扩大和 deterministic merge；
- new master 无任何历史 KV 时，DLSLIME remote attention 正常执行；
- 多轮 master 变化后无 block leak、负计数或 stale mask；
- EP32 collective ordering 不因某 DP/master batch 为空而死锁；
- historical KV migration bytes 始终为 0；
- feature flag 关闭时性能与行为保持原基线。

Dummy Prefill 集成测试只要求 scheduler、shape、block table、通信和 CUDA kernels 稳定运行，不验证语言输出质量或真实 Prefill 数值结果。

### 15.3 性能验收

正式启用 batch threshold 前必须完成：

```text
batch_size x context_length x master_dop x kv_dop
```

profiling，并满足：

- 存在可重复的 single-master/multi-master crossover；
- `T_compute` 来自独立 calibration workload；
- scale-up 后 ITL 收益覆盖新增 SP 通信和同步成本；
- master batches 与 source-side dispatch/combine 负载合理分散；
- scheduler overhead 相对 Decode ITL 足够小；
- 不增加 KV capacity preemption；
- 至少报告 threshold 的 ±25% 和 ±50% sensitivity。

如果 profiling 证明固定 EP32 下增加 masters 没有收益，则停止 compute-triggered scale-up，只保留 memory-triggered multi-master，不能为了匹配 LoongServe 机制而强行启用。

## 16. 实施步骤

### Phase 1：Profiling 与 Shadow Planning

- profiling 1/2/4/8 masters 的 Decode scaling curve；
- 确定或否定 compute-bound crossover；
- shadow 生成 pack-full-first、source-style threshold chunk master plan，不提交；
- shadow 生成自动/强制 `D_init` 和 uniform prompt placement；
- 验证 batch、capacity、initial placement 和 deterministic assignment。

### Phase 2：Iteration-Scoped Multi-Master

- 实现 pending-token frontier；
- 实现 batch-level `D_init`、immutable initial placement records 与 uniform prompt placement；
- 实现 plan/validate/commit；
- 每轮使用 source-style threshold greedy 重算 masters；
- 支持 participant reuse 和 no-history-migration scale-up；
- 修正 zero-history master 的 SP/data-plane metadata。

### Phase 3：Multi-Group 与实验闭环

- 实现 group allocation 扩大和 deterministic merge；
- 完成指标、单元测试和集成测试；
- 跑 fixed master DoP=1/2/4/8、oracle static DoP 和 LS-Decode-Core；
- 固化 calibration、benchmark 和 plot scripts。

Core 实现到 Phase 3 结束。历史 KV rebalance、rank-releasing scale-down 和 runtime KV relayout 不属于本文档。

## 17. 实现前确认项

1. Baseline 名称为 `LS-Decode-Core` 或 `LoongServe-style decode-only scheduler`，不称完整 LoongServe。
2. 目标模型为 DeepSeek-V3，EP32 始终固定。
3. master 是 real token 的 attention master 和 MoE dispatch/combine source，不是独立完整模型 replica。
4. `loop_count=1` 是硬性支持边界；masters 在每个 token iteration 重算。
5. `D_init` 是 batch-level initial KV DoP；所有 requests 在相同 initial ranks 上均匀切分 prompt KV。
6. master rank 采用 pack-full-first，不使用 sticky preference。
7. master slices 使用 LoongServe source-style threshold greedy，不使用均衡 `ceil(remaining / masters)` 切片。
8. compute/memory scale-up 都不迁移历史 KV。
9. batch 下降只减少 active master roles；`D_init` 不变，仍持有 KV 时 `kv_dop` 不下降，不做 KV evacuation。
10. 不实现 hysteresis、cooldown、scale-up rebalance 或 rank-releasing migration。
11. 复用现有 NanoDeploy/DLSLIME data plane，不复刻 LoongServe overlapped kernel。
12. Dummy Prefill 只用于 Decode 性能，不验证语言输出或真实 Prefill 数值语义。
