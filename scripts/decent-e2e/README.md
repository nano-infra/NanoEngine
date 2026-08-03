# Decentralized E2E: Long-Short DP4/SP8/EP32

入口脚本：`run_decent_e2e_longshort_dp4sp8_ep32.sh`。

## 配置基线

这组脚本组合了两套已经跑过的基线：

- workload 与 2026-04 的中心化 E2E 对齐：每个 rate 的到达窗口为 600 秒，batch size 256，`max_model_len`、`max_input_len`、`max_request_tokens` 都是 1,000,000；请求数严格为 `rate * 600`。
- 去中心化控制面采用 July 最近完整成功实验的配置：`scheduler_arch=hierarchical`、`router_policy=least_batch_v2`、SP master 选 `LeastBatch`、`hao_basic`、full CUDA graph、ZMQ worker transport、`SLIME_QP_NUM=4`。
- Long-Short 为 `long_short_sp8`：prompt 长度不超过 100,000 时使用 SP1，超过 100,000 时使用 SP8。
- 拓扑为 attention DP4 × SP8 × TP1，共 32 GPU；FFN 为 DP1 × EP32 × TP1。当前 RayExecutor 每节点固定放 8 个 worker，因此需要 4 个空闲的 8-GPU 节点。

与旧 E2E 相同的引擎容量常量由当前 benchmark 固定：

| 参数 | 值 |
| --- | ---: |
| `max_num_seqs` | 256 |
| `max_num_batched_tokens` | 1,024,000 |
| `kvcache_block_size` | 64 |
| `max_num_recv_seqs` | 32 |
| warmup requests | 256 |
| `segment_size` | 65,536 |
| `loop_count` | 16 |
| GPU memory limit/utilization | 141 GiB / 0.9 |

注意：benchmark 里保留了 `max_num_send_seqs=16` 这个调用参数，但旧版和 July 版 `LLMEngine` 都只接受 `Config` dataclass 中存在的字段，而 `max_num_send_seqs` 不在其中，所以它在旧实验里也是 no-op。真正生效并与旧实验对齐的是 `max_num_recv_seqs=32`；脚本会检查源码中的这些常量，若以后被改动则拒绝启动，要求重新审计。

ZMQ 在 July 仓库中仍是 opt-in，项目全局默认 transport 仍为 Ray。本脚本之所以显式默认 ZMQ，是因为当前最后一轮完整成功日志就是 ZMQ 配置；若这次矩阵希望走已发布的保守默认，只需在启动命令前加 `NANODEPLOY_HIER_WORKER_TRANSPORT=ray`，其他去中心化参数不变。

## Rate 矩阵

| 数据集 | Rates | 600 秒对应请求数 |
| --- | --- | --- |
| issue 1% | 10, 20, 30, 35, 40, 50, 60, 70, 80, 90 | 6k, 12k, 18k, 21k, 24k, 30k, 36k, 42k, 48k, 54k |
| issue 5% | 5, 10, 20, 30, 40, 45 | 3k, 6k, 12k, 18k, 24k, 27k |

600 秒是请求到达窗口；总墙钟时间还包括 warmup、长请求尾部 drain、清理及两轮之间的冷却，因此会大于 600 秒。

## 启动

先确认 Ray 集群为空闲状态，并且 head 节点 IP 是 `10.102.234.33`。脚本默认使用 Ray `10.102.234.33:7789`；MASTER 延续旧实验端口 `27799`，即 `10.102.234.33:27799`。

```bash
cd /mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-July

RUN_TAG=decent_e2e_issue001_issue005_longshort_dp4sp8_ep32_bs256_$(date -u +%Y%m%d_%H%M%S) \
  ./scripts/decent-e2e/run_decent_e2e_longshort_dp4sp8_ep32.sh
```

只跑一个数据集：

```bash
RUN_TAG=decent_e2e_issue001_smoke \
  ISSUE001_RATES="10" \
  ./scripts/decent-e2e/run_decent_e2e_longshort_dp4sp8_ep32.sh issue001
```

只检查将要执行的命令，不连接 Ray/GPU：

```bash
DRY_RUN=1 RUN_TAG=decent_e2e_dryrun \
  ./scripts/decent-e2e/run_decent_e2e_longshort_dp4sp8_ep32.sh
```

日志默认写到 `bench_logs/decent-e2e/$RUN_TAG`。每轮只有在 `.summary.json` 请求计数、transport 和日志末尾 `BENCH_RESULT` 的发送/接收/拒绝/残留请求计数全部通过校验后，才会写 `SUCCESS`。相同 `RUN_TAG` 重启时，脚本会核对 `effective_config.tsv` 并跳过已有成功标记的 stage；若有效配置发生变化，会要求换新的 `RUN_TAG`，避免把不同实验拼进同一组结果。

脚本默认在单个 rate 连续失败 5 次后停止整组实验。需要继续后续 rate 时可设置 `CONTINUE_ON_ERROR=1`；需要重跑已有成功项时设置 `FORCE_RERUN=1`。
