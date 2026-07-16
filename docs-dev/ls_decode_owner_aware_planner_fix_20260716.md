# LS-Decode owner-aware planner 修复与单机验证

日期：2026-07-16

## 结论

已修复 T=128 两机压测中触发连续抢占的调度假阴性。原实现把 request admission order 当成连续 prefix 的硬约束；当 receiver 接近上限时，greedy prefix 可能失败，但同一组请求仍存在合法的 owner-local assignment。修复后，source-greedy 仍作为热路径；只有失败时才进入 owner repair、sticky repair 和 receiver/append flow fallback。

本次只修改 NanoDeploy。相关 C++ 扩展已重新安装，相关 CPU 回归 93/93 通过；最终二进制完成单机 8 GPU 压力验证和最终 smoke，均为 0 planner failure、0 preemption、0 Traceback/RuntimeError。

## 修复内容

### 1. 保留热路径，增加稳定 owner repair

`plan_iteration_masters_source_greedy` 首先执行原有 source-greedy。失败后按 candidate/current master 稳定分桶，并把 assignment 映射回原 request index。该路径直接覆盖真实 443-request 事故的结构：原 admission-order prefix 失败，而 owner bucket 顺序可形成合法 chunks。

如果当前 master placement 本身仍满足权威 validator，则可走 `sticky_repair`，避免 group merge 仅因顺序变化破坏原有合法 placement。

### 2. receiver 约束改成 owner-local quota

对 rank `r`：

```text
owner_count[r] = 持有该 rank 历史 KV 的请求数
lower[r]       = max(0, owner_count[r] - max_num_recv_seqs)
```

`lower[r]` 是必须以 rank `r` 为 master 的 owner-local 请求数。fallback 使用 capacity max-flow 检查这些 quota，包括非平凡 Hall violation；required owner 只在 extra ranks 中时，会先激活该指定 rank 并 restart，而不是误报不可行。

### 3. receiver、append、metadata 联合求解

对正常 KV block size（`>=2`），一次 pending append 的额外 block cost 为 0 或 1，且 cost 0 必然是 owner。固定 load 下按 rank 建三类 flow capacity：

- 必需 cheap-owner；
- 其余必需 owner；
- general。

这样 receiver lower quota、append block budget 和 master metadata capacity 在同一个 assignment flow 中满足，避免此前 worst-prefix / worst-subset 安全包络造成新的假阴性。

如果 source-style nominal load vector 本身不可行，则进入完备的 variable-load branch-and-flow。flow 先在每个 rank 的 load upper bound 内自行选取 load；若某 rank 超过 append budget，所有合法解必然落入两个分支之一：

```text
A: 该 rank 的 load 至少减少 1
B: 该 rank 强制提供满足 block budget 所需的 cost-0/cheap owner 数
```

两个分支都严格收紧整数边界，搜索队列穷尽后才会报告 joint capacity proven infeasible，不再使用会耗尽后误抢占的有限 load-vector 枚举。新增回归既覆盖 `[2,2,0]` 不可行、但 `[2,1,1]` 可行的近邻场景，也覆盖旧的 128 次枚举耗尽、合法 load 位于 `[0,6,10,11]` 的远距离场景。

### 4. 计划一致性与诊断

`LSDecodeMasterPlan` 新增 `assignment_strategy`，可区分：

- `source_greedy`；
- `owner_bucket_repair`；
- `sticky_repair`；
- `receiver_append_flow`；
- `receiver_append_flow_load_repair`。

validator 现在同时检查：master rank/count telemetry 长度、rank 唯一性、正 batch size、总数覆盖所有请求，以及 declared count 与逐请求 assignment 一致。

失败原因也拆分为 receiver proven infeasible、decode metadata、append capacity 和 joint search 等类别，避免把所有失败都笼统解释成 KV blocks。

## 回归测试

执行：

```bash
pytest -q \
  tests/test_ls_decode_planner.py \
  tests/test_ls_decode_scheduler.py \
  tests/test_ls_decode_config.py \
  tests/test_ls_decode_metadata.py \
  tests/test_ls_kv_scale_down.py
```

结果：`93 passed in 2.20s`。

新增/强化的关键用例包括：

- admission-order prefix 假阴性，owner bucket repair 成功且不无谓扩 rank；
- receiver quota 总量不可行与非平凡 Hall violation；
- required owner 位于 extra rank；
- decode metadata 与 receiver/append 原因分离；
- 互补 cost-0 append 子集；
- nominal load vector 不可行、相邻 load vector 可行；
- 旧有界搜索耗尽但远距离 load vector 可行；
- `reserved_blocks_per_req=1.0` 下 exact-flow、validator 和实际 plan commit；
- 拒绝负数 `reserved_blocks_per_req`，保证 branch 单调性前提；
- plan telemetry 与 assignment 一致性。

## 单机 8 GPU 压力验证

配置：DP1×SP8 / EP8，`max_num_seqs=256`，receiver/T=`64/64`，prompt/output=`4096/256`，rate 8，60 秒发送窗口，最多 256 inflight，consolidation=`execute`，eager，dummy weight，并开启 actual-forward CUDA event timing。

结果：

- accepted/completed：`256/256`；unfinished 0；
- 60 秒发送 + 61.11 秒 drain，drain completed；
- max inflight 256，最大 KV util 61.68%；
- planner failure 0，preemption 0；
- planning latency：p50 0.225 ms，p95 0.335 ms，max 0.443 ms；
- actual forward：437 samples，p50 263.01 ms，p95 279.78 ms；首次 warmup max 1293.93 ms；
- 无 CUDA/OOM/Traceback/RuntimeError。

产物：

- `docs-dev/ls_decode_owner_repair_8gpu_scaled_r8_60s_20260716.log`
- `docs-dev/ls_decode_owner_repair_8gpu_scaled_r8_60s_20260716.json`
- `docs-dev/ls_decode_owner_repair_8gpu_scaled_r8_60s_20260716.jsonl`

该 workload 把 256 个请求分布到 8 个单-owner group，因此验证了高并发和最终二进制稳定性，但没有重放两机日志中的 443-request group merge；该假阴性由确定性 planner regression 直接覆盖。

## 最终 branch-and-flow 二进制 smoke

完备 branch-and-flow 修改重新编译后，再运行 8 秒 DP1×SP8 smoke：

- accepted/completed：`17/17`；
- drain completed，unfinished 0；
- planner failure 0，preemption 0；
- 0 Traceback/RuntimeError。

产物：

- `docs-dev/ls_decode_owner_repair_branch_flow_final_8gpu_smoke_20260716.log`
- `docs-dev/ls_decode_owner_repair_branch_flow_final_8gpu_smoke_20260716.json`
- `docs-dev/ls_decode_owner_repair_branch_flow_final_8gpu_smoke_20260716.jsonl`

## 两机验收重点

建议沿用原 Issue 1% CSV、DP2×SP8、rate 20、360 秒、T128/receiver128、consolidation execute 配置。重点检查：

1. 原约 250 秒的首次 `plan failed` 不再出现；
2. `Preemption happens` 和 recovery admission 循环为 0；
3. 443-request merge 附近 planning latency、receiver counts 和 master chunks 合法；
4. actual-forward 与 step ITL 同时保留，区分模型耗时和排队/调度耗时；
5. 发送窗口结束后可自然 drain 并生成完整结果。

variable-load fallback 只在 source-greedy、owner bucket、sticky 和固定-load flow 均失败时运行；典型路径不承担额外搜索成本。其正确性搜索最坏状态空间仍可能较大，所以两机验收还应重点观察 `planning_latency_ms` 尖峰。新增远距离反例的完整构造与求解单测约 10 ms；后续若发现真实 workload 的 branch 状态数较大，应增加按 group state fingerprint 保存 search frontier 的分步预算机制，但预算耗尽不得重新解释为不可行或直接抢占。

如果两机测试仍出现失败，应保留首次失败前后完整 `ls_decode_iteration`、plan failure reason 和 actual-forward 行，优先判断它属于 receiver proven infeasible、真实 append/KV pressure，还是 joint search exhaustion；不要再只根据 `receiver=128` 推断 KV cache 已满。
