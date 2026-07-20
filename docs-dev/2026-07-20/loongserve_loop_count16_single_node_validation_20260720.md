# LoongServe-style `loop_count=16` 单机验证

时间：2026-07-20 UTC

## 结论

当前 NanoDeploy LoongServe-style Decode 路径已经可以安全配置
`loop_count=16`。调度器把配置值作为每轮上限，并按所有本轮 RUNNING
请求中最短的剩余输出长度缩短尾轮。因此 `max_tokens=34`（其中 admission
bootstrap 占 1 token）的实测 Decode chunk 为 `16 + 16 + 1`，不会超生成。

在 Ray `10.102.252.174:6380` 的单节点 8×H200 上，8、64、128 请求三组
测试都 exit 0，所有请求完成且输出长度精确匹配。

## DoP 与扩缩容结果

物理并行拓扑固定为：

- Attention：DP=1、SP=8、TP=1
- FFN：DP=1、EP=8、TP=1

LoongServe-style group 内的动态 DoP 结果：

| workload | iteration master DoP | KV DoP | 扩缩容 |
|---|---:|---:|---|
| 8 requests | 1 | 1 | 无 |
| 64 requests | 1 | 1 | 无 |
| 128 requests，首轮 | 2 | 1 | compute scale-up：新增 rank 7 |
| 128 requests，后续轮 | 2 | 2 | 新 rank 已承载 KV |

128 请求首轮的 master ranks 为 `[0, 7]`，batch split 为 `[100, 28]`，
日志中的 `scale_reason` 为 `compute`。这次同长度 workload 没有出现 KV
scale-down：128 个请求始终同时存活，随后一起结束，group 直接做 zero-live
cleanup，没有低负载长尾窗口可供 consolidation。

需要特别注意：正式 source-default profile 的 compute-bound threshold 是
`ls_min_comp_bound_decoding_batch_size=100`。比较脚本保留的
`--ls-batch-per-master=8` 会进入 manifest，但当前正式调度路径的 scale-up
判断使用的是前者。因此 64 请求不会扩容，128 请求只扩到 DoP=2，而不是 8。

## 实现要点

- `Config` 的 LoongServe-style 合法范围改为 `loop_count in [1, 16]`；C++
  scheduler 还要求 chunk 小于 KV block size（正式配置为 64）。
- `ScheduleResult.execution_loop_count` 携带本轮实际 K；Ray executor 将同一个
  K 传到所有 workers，model runner 恰好执行 K 次 forward。
- KV planner、validator 和 prepared transaction 按
  `committed + current_pending + K_outputs` 预留 block，覆盖 chunk 跨 KV block
  边界的情况。
- engine 的 postprocess、token accounting、ITL 和日志均使用实际 K；尾轮可从
  16 自动降为 1。

## 验证

修改 C++ 后已执行并成功完成 `pip install -v -e .`。

CPU/C++ 回归：

```text
99 passed
101 passed
total: 200 passed
```

其中独立 prepared-iteration C++ harness 为 7/7，通过了 prompt=63、block=64、
K=16 的跨 block reservation 测试。

GPU 命令的共同关键参数：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
python scripts/bench_ls_decode_itl_compare.py \
  --case ls \
  --ray-address 10.102.252.174:6380 \
  --num-requests <8|64|128> \
  --prompt-len <4096|512|512> \
  --max-tokens 34 \
  --loop-count 16 \
  --max-steps 12
```

产物：

- `ls_decode_chunked16_single_node_8req_20260720.json`
- `ls_decode_chunked16_single_node_scale_64req_20260720.json`
- `ls_decode_chunked16_single_node_scale_128req_20260720.json`

三份 JSON 的 Decode loop 都是 `[16, 16, 1]`；最终完成数分别为 8、64、128。
