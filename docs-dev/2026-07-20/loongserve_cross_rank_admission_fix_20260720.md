# LoongServe-style 跨 rank admission 修复记录

日期：2026-07-20 UTC

## 结论

`gpu_memory_utilization=0.85` 的双节点 DP2×SP8 正式 workload 在约 21 秒后报
`UnschedulableRequestError: request cannot fit the full empty SP pool exactly`，
根因是初始 prompt placement 在打包前没有扣除 master bootstrap token 和
`reserved_blocks_per_req`。总容量足够的长请求一旦跨 rank，就会被错误拒绝。

入口空系统预检和正式 admission planner 均已修复。对应的 DP>1 纯 CPU 回归、
原始触发样本 CPU 复现和双节点真实 GPU smoke 均通过。

## 触发样本与容量

失败 run 的 manifest 记录：

```text
attention_dp=2
attention_sp=8
loop_count=16
gpu_memory_utilization=0.85
num_kvcache_blocks_per_rank=12372
resolved_admission_max_tokens_per_pool=1055744
```

workload 中零基第 441 条、CSV 第 443 行是：

```text
prompt_len=811259, output_len=754, total_len=812013
```

每个 SP manager 的常驻 cadence dummy 会占 1 个 block，因此该 run 中空系统
留给请求的每-rank block 数是 `12371`。在 master 上还必须保留 1 个 reserved
block，并容纳 1 个 bootstrap token，所以单 rank 可打包的 prompt 上限实际为：

```text
(12371 - 1) * 64 - 1 = 791679 tokens
```

`811259 > 791679`，该请求必须使用至少两个 KV rank。这不是 pool 总容量不足。

旧代码先按原始 free-token 容量把 rank0 填满，再在精确校验阶段追加 bootstrap
和 reserved headroom。rank0 因而必然超限；增加候选 DoP 后，打包器仍会先填满
rank0，所以从 DoP=1 一直错误失败到 DoP=8。

## 修复

对每个候选初始 DoP，先确定已有和新增 master load，然后在 prompt 打包前从
每个 rank 的容量中扣除：

1. `ceil(total_master_load * reserved_blocks_per_req)` 个 reserved block；
2. 每个新增 master 的 1 个 bootstrap token。

打包后的逐请求 block rounding 和精确容量校验仍然保留，因此该修改只消除
false negative，不会允许超容量 placement。

相同逻辑同时应用于：

- `Scheduler::_ls_batch_fits_empty_system`：`add()` 的永久可调度性预检；
- `Scheduler::_plan_ls_initial_placement`：实际 admission / capacity-append 规划。

## 无 GPU 回归

新增 DP2×SP2、block size 64、每 rank 4 blocks、reserved=1、K=16 的缩小等价
用例。270-token prompt 必须跨两个 rank：

- 旧实现：`scheduler.add()` 返回 `CURRENT_EXACT_NO_FIT`；
- 新实现：成功接纳，`planned_kv_dop=2`、ranks `[0,1]`；
- bootstrap 后 dispatched tokens 为 `[128,143]`；
- 两个 rank 分别分配 `[2,3]` blocks，master 仍保留 reserved block。

使用失败 run 的精确容量和原始 811259/754 样本做 CPU 复现，新实现结果为：

```text
accepted=True
error=LSAddError.NONE
planned_kv_dop=2
ranks=[0,1]
```

首 7200 条中的最大 prompt `971548` 也成功以 DoP=2 接纳。相关 CPU 回归两组
分别为 `104 passed` 和 `97 passed`，合计 `201 passed`。

## 双节点 GPU 验证

在 Ray `10.102.243.60:8776` 上运行 DP2×SP8、`.85`、K=16 的单请求 smoke，
保留原始 `prompt_len=811259`，将输出缩短到 17 token，以覆盖跨-rank admission
和一次完整 K=16 Decode：

```text
status=success
requests_sent=1
requests_completed=1
prompt_len=811259
output_len=17
ITL samples=16
total_time_sec=2.4524408818688244
```

产物：

- `ls_style_loop16_2node_cross_rank_811k_smoke_20260720.manifest.json`
- `ls_style_loop16_2node_cross_rank_811k_smoke_20260720.jsonl`

benchmark 写盘并 exit 0 后出现的 TCPStore connection-reset 是 Ray actor teardown
期间 rank0 先退出产生的告警，不影响完成结果。退出后 Ray 为 `0.0/16.0 GPU`、
无 pending demand，本机也没有残留 CUDA 进程。
