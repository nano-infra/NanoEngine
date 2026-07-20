# LoongServe-style loop_count=16 双节点 idle-DP 修复记录

日期：2026-07-20 UTC

## 结论

双节点 `attention_dp=2`、`attention_sp=8`、`loop_count=16` 在首个请求仅落到
DP0 时触发的 `IndexError: Block index out of range` 已修复。修复后的同拓扑
真实 GPU 稀疏启动 smoke 完成 `1 / 1` 请求并 exit 0，manifest 状态为
`success`。

该异常不是 K=16 跨 KV block 的容量预留错误，而是空闲 DP 的 dummy
序列缺失 KV block table。

## 根因

正式 workload 使用 `arrival_round_robin`。第一个请求到达时只有 DP0 有真实
Decode group，DP1 暂时为空。组合调度器原先仅在“当前 DP 自己存在真实 Decode
group”时，才给该 DP 内没有真实负载的 SP rank 添加持久化 dummy：

```cpp
master_load[rank] == 0 && !step->decode_by_dp[dp_idx].groups.empty()
```

因此 DP1 的 8 个 rank 收到空 `dp_seqs`。`ModelRunner` 面对空列表时会临时构造
一个本地 dummy `Sequence`，但这个对象没有经过调度器的 KV 分配，block table
为空。`prepare_decode_cpp` 读取最后一个 block page id 时即抛出越界异常。原日志
中的全局 rank 8 正是 DP1/SP0。

两套 DP pool 的 attention/KV 拓扑彼此独立，但 FFN cadence 全局同步。因此只要
本轮任意 DP 正在 Decode，空闲 DP 也必须收到可执行的、已经分配 KV 的 dummy。

## 修复

在组合调度器中先计算本轮是否存在任意真实 Decode：

```cpp
const bool has_global_decode = std::any_of(
    step->decode_by_dp.begin(), step->decode_by_dp.end(), [](const auto& pending_dp) {
        return !pending_dp.groups.empty();
    });
```

当 `has_global_decode` 为真时，所有 DP 内没有真实负载的 SP rank 都使用调度器
维护且已分配 KV block 的持久化 rank dummy。纯 admission 步骤仍不会无条件添加
dummy。

新增回归测试构造 DP2×SP8、block size 64、K=16，仅向 DP0 添加一个请求，并
验证：

- DP0 包含唯一真实请求，DP1 没有真实请求；
- 两个 DP 的 `dp_seqs` 都包含 8 个 rank entry；
- DP1 的 8 个 dummy 分别覆盖 master SP rank 0..7；
- 每个 dummy 的 active block context 都存在有效 page id。

## 验证

修改 C++ 后按仓库要求执行了：

```bash
pip install -v -e .
```

相关 CPU 回归分两组运行，分别为 `100 passed` 和 `101 passed`，合计
`201 passed`。

双节点 GPU smoke 使用 Ray `10.102.243.60:8776`，关键配置如下：

```text
attention_dp=2
attention_sp=8
ffn_ep=16
loop_count=16
prompt_len=221
max_tokens=17
num_requests=1
cuda_graph_mode=full
```

这个单请求场景刻意保持 DP1 空闲，可稳定覆盖原来出错的 DP1/SP0 路径。结果：

```text
status=success
requests_sent=1
requests_completed=1
total_time_sec=2.1223620511591434
output_len=17
ITL samples=16
```

`output_len=17` 且有 16 个 ITL 样本，符合 bootstrap token 后执行一次 K=16 Decode
的预期。

产物：

- `ls_style_loop16_2node_idle_dp_smoke_20260720.manifest.json`
- `ls_style_loop16_2node_idle_dp_smoke_20260720.jsonl`

退出后 Ray 状态为 `0.0/16.0 GPU`、无 pending demand，本机
`nvidia-smi --query-compute-apps` 也没有残留进程。日志中偶发的 Ray metrics
exporter 连接告警仍可能出现，但它与本次 block table 越界无关，也不影响请求完成。
