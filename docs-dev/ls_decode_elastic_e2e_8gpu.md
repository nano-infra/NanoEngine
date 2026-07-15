# LS Decode 动态负载 8-GPU E2E

日期：2026-07-15

## 目标

在单机 `Attention DP1 × SP8 × TP1 / FFN EP8` 上验证同一个 Decode
group 的完整状态闭环：

```text
burst 1: 64 requests / master_dop=8 / kv_dop=8
    -> arrival-idle: 1 anchor / automatic consolidation 8->1
    -> burst 2: merge into the same group / master_dop=8 / kv_dop=8
    -> drain
```

这里的中间阶段是“停止新请求到达，但保留一个长尾请求继续 Decode”，而不是
零 inflight 的 wall-clock idle。完全空闲时不会再调用 `engine.step()`，活跃 group
也会随请求完成而销毁，因此不存在可供自动 consolidation 的 live KV。

## 配置

- DeepSeek-V3，dummy weight、dummy prefill；
- Full CUDA Graph，每个 worker 捕获 8 个 local graphs 和 72 个 SP graphs；
- 第一波：1 个 `max_tokens=32` anchor + 63 个 `max_tokens=8` 请求；
- 第二波：63 个 `max_tokens=8` 请求；
- `prompt_len=4096`，`ls_decode_initial_kv_dop=8`；
- `ls_decode_batch_per_master=8`；
- KV consolidation `execute`；
- correctness 快速阈值：`stable_steps=1`、`cooldown_steps=0`、
  `check_interval_steps=1`；
- `max_source_blocks_per_event=16`，`migration_chunk_tokens=64`。

运行命令：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -u scripts/test_ls_decode_elastic_e2e_8gpu.py \
  --ray-address 10.102.243.60:8776 \
  --master-address 10.102.243.60:29776 \
  --output-json docs-dev/ls_decode_elastic_e2e_8gpu.json
```

## 结果

测试通过，结构化断言确认：

1. 第一波在 group 0 达到 `real_batch=64 / master_dop=8 / kv_dop=8`，master
   slices 为 `[8,8,8,8,8,8,8,8]`；
2. 短请求完成后，仅剩一个 anchor；执行 7 次真实 KV P2P maintenance，rank
   allocation 依次 `8->7->6->5->4->3->2->1`；
3. 7 次 maintenance 分别耗时约 `92.7/108.6/121.0/114.8/99.9/129.9/144.7 ms`；
4. arrival-idle 窗口稳定达到 `real_batch=1 / master_dop=1 / kv_dop=1`；
5. 第二波 admission kind 为 `merge`，仍进入 group 0；下一次 Decode 恢复为
   `real_batch=64 / master_dop=8 / kv_dop=8`；
6. 两次高负载 Decode telemetry 的 `historical_kv_migration_bytes=0`；
7. 总共 40 个 engine steps、127 个请求全部完成，输出长度均符合请求配置；
8. 8 个 Ray ModelRunner workers 测试结束后全部正常退出。

详细结果见 `docs-dev/ls_decode_elastic_e2e_8gpu.json`。Ray driver 输出过一次 metrics
exporter agent 连接告警，但不影响 worker、CUDA Graph、NCCL、KV transaction 或请求完成。

## 边界

这是 dummy-weight、dummy-prefill 的 correctness E2E，不是性能结论。快速
consolidation 阈值用于在短测试中覆盖完整状态转换，不代表生产 hysteresis/cooldown
配置；该测试也只覆盖一个 DP1×SP8/EP8 域，不能替代 DP4×SP8/EP32 验收。
