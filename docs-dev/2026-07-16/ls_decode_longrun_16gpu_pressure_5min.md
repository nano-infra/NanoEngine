# LS-Decode-Core 两机 16-GPU Consolidation 5 分钟压力测试

日期：2026-07-16

## 环境与配置

- NanoDeploy commit：`b734fd3`（`feat: support two-node LS decode validation`）
- Ray GCS：`10.102.243.60:8776`
- 节点：`10.102.243.60`、`10.102.206.14`
- 资源：2 节点，16 × H200
- topology：Attention `DP2 × SP8 × TP1`，FFN `EP16`
- model：DeepSeek-V3，dummy weight，dummy prefill
- execution：full CUDA Graph；每个 worker 捕获 8 个 local graphs 和 72 个 SP graphs
- workload：300 秒发送窗口，request rate `8 req/s`，burstiness `6`，prompt `4096` tokens，output `128` tokens
- LS scheduler：`initial_kv_dop=1`，`batch_per_master=8`，memory scale-up enabled
- consolidation：`execute`，candidate util `0.50`，target high watermark `0.80`，stable/cooldown/check interval steps `2/2/1`
- migration：每事件最多 128 source blocks，chunk size 64 tokens

本轮沿用单机 pressure profile，并将 request rate 和 inflight 上限按两机容量扩大为两倍。代理环境变量在连接 Ray 前已清除。

## 结果

- 进程正常退出，exit code 为 0；300 秒发送窗口结束后在约 `12.79 s` 内完成 drain。
- 接受并完成 `2051/2051` 个请求；0 backpressure drops、0 unfinished requests；最大 inflight 为 233。
- 完成 `225` 个 admission/prefill control steps、`2987` 个 Decode steps 和 `90` 个真实 consolidation maintenance steps。
- 90 次 maintenance 总 wall time 约 `6028.01 ms`；mean/median 约 `66.98/34.36 ms`，p90 约 `153.04 ms`，最大约 `330.73 ms`。其中 85 次发生在发送窗口，5 次发生在 drain。
- 请求吞吐为 `6.556 req/s`，输出吞吐为 `839.184 token/s`；两者使用包含 drain 的总 wall time `312.84 s` 计算。
- request average ITL：mean `95.29 ms`，median `94.67 ms`，p99 `104.46 ms`。
- steady Decode step ITL（包含前置 maintenance stall）：median `100.63 ms`，p95 `107.58 ms`，p99 `162.77 ms`，最大 `679.39 ms`。
- E2E latency：median `12.876 s`，p95 `15.293 s`，p99 `16.000 s`。
- 未观察到 Ray actor、NCCL、CUDA Graph、DLSLIME、KV P2P、scheduler transaction 或 block ownership 错误。
- 结束后 16 个 ModelRunner actors 均已终止；Ray 报告两节点仍 active，CPU/GPU 使用均为 0，pending demands 为空。

## 结论边界

本轮验证了 LoongServe-style multi-master Decode scheduler 在 `DP2 × SP8 / EP16` 两机拓扑下，能够在 bursty workload 中反复执行 KV consolidation、再次 scale-up，并连续运行 5 分钟后完整 drain。它使用 dummy weight 和 dummy prefill，且 consolidation 时间阈值比默认值更激进，因此是稳定性与执行链路验证，不是生产容量结论，也不是完整 LoongServe 复现。

原始结果：

- `ls_decode_longrun_16gpu_pressure_5min.json`
- `ls_decode_longrun_16gpu_pressure_5min.jsonl`
