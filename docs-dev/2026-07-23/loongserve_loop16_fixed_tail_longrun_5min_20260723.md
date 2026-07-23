# LoongServe-style `loop_count=16` 固定尾轮单机 5 分钟长跑

时间：2026-07-23 03:34--03:41 UTC

## 结论

固定 `loop_count=16` 修改通过单机 8×H200 的 5 分钟持续验证。

- 连续注入 300 秒；
- 6.61 秒完成 drain；
- 1,264 个 accepted request 全部完成；
- 90 个 Decode dispatch 的 scheduler execution K 全部为 16；
- 90 个 Decode dispatch 的 worker CUDA-event 聚合 K 全部为 16；
- 1,264 个请求的可见输出长度全部精确为 18；
- manifest 为 `success`，进程 exit 0。

这说明修改不仅在两轮 smoke 中生效，也能在持续 admission、batch
变化和 master rank 轮换的长跑中保持固定执行 cadence。尾轮多执行的
15 个 token slot 会被 postprocess 丢弃，不会泄漏到用户可见输出。

## 前置入口扩展

原 `scripts/bench_ls_decode_longrun_8gpu.py` 把 `loop_count` 固定为 1，
无法用于本次 K=16 长跑。为此增加了：

- `--loop-count`，合法范围 `[1, 16]`，默认仍为 1；
- engine 和 resolved manifest 使用同一个配置 K；
- 每个 Decode step 记录实际 `execution_loop_count`；
- step ITL 按实际 K 折算。

CPU 验证：

```text
python -m py_compile scripts/bench_ls_decode_longrun_8gpu.py
PYTHONMALLOC=debug pytest -q tests/test_ls_decode_benchmark_profile.py
5 passed in 2.29s
```

实现提交：

```text
23db1fd feat: support chunked LS longrun tests
```

没有修改 C++，因此不需要重新执行 editable reinstall。

## 环境和命令

- Ray：`10.102.206.14:7789`
- master endpoint：`10.102.206.14:29906`
- topology：Attention DP1×SP8×TP1，FFN EP8
- GPU：8×NVIDIA H200
- backend：`hao_basic`
- eager execution
- `dummy_prefill=true`、`dummy_weight=true`
- prompt length：63
- `max_tokens=18`
- `loop_count=16`
- request rate：4 req/s
- burstiness：2.0
- inflight cap：64

命令：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  NANODEPLOY_LOG_MODEL_FORWARD_TIMING=1 \
  PYTHONUNBUFFERED=1 \
  python scripts/bench_ls_decode_longrun_8gpu.py \
  --ray-address 10.102.206.14:7789 \
  --master-address 10.102.206.14:29906 \
  --duration-sec 300 \
  --drain-timeout-sec 120 \
  --request-rate 4 \
  --burstiness 2.0 \
  --max-inflight-requests 64 \
  --prompt-len 63 \
  --max-tokens 18 \
  --max-num-seqs 64 \
  --max-num-recv-seqs 128 \
  --attention-dp 1 \
  --loop-count 16 \
  --steady-start-sec 60 \
  --progress-interval-sec 30 \
  --ls-max-num-ooe 8 \
  --output-json \
    docs-dev/2026-07-23/ls_loop16_fixed_tail_longrun_8gpu_5min_20260723.json \
  --completion-jsonl \
    docs-dev/2026-07-23/ls_loop16_fixed_tail_longrun_8gpu_5min_20260723.jsonl \
  --manifest-json \
    docs-dev/2026-07-23/ls_loop16_fixed_tail_longrun_8gpu_5min_20260723.manifest.json
```

Ray/NanoDeploy 运行前清除了四个 HTTP(S) proxy 变量。

## 结果

### 完成性

```text
manifest status:               success
total elapsed:                 306.898 s
send duration:                 300.000 s
drain duration:                  6.612 s
accepted requests:                1264
completed requests:               1264
unfinished requests:                  0
drain completed:                   true
max inflight observed:               64
backpressure dropped arrivals:       35
```

35 个 dropped arrivals 是到达 burst 碰到 inflight cap 后、尚未 admission
就被脚本拒绝的负载保护事件。所有 1,264 个 accepted requests 都成功完成。

### 固定 K 和输出正确性

```text
step count:                         91
decode steps:                       90
maintenance steps:                   1
scheduler execution K histogram: {16: 90}
worker timing K=16 records:          90
completion records:               1264
completion output length:      {18: 1264}
```

每个请求 admission bootstrap 先产生 1 token。第一轮 Decode 保留 16 个
有效 token，尾轮只剩 1 个有效 token，但仍执行 16 次 forward。因此每个请求
都反复覆盖本次修复的关键固定尾轮，而可见输出仍严格等于 18。

日志中没有：

- inconsistent model-forward loop warning；
- `LSSchedulerFatal`；
- Python traceback；
- CUDA error；
- NCCL error；
- drain timeout。

Ray metrics exporter 的 observability warning 不影响执行和结果。

### 性能观测

```text
request throughput:               4.119 req/s
output throughput:               74.135 token/s
steady Decode ITL mean:         208.356 ms/token
steady Decode ITL p99:          211.430 ms/token
steady Decode ITL min/max:      204.335 / 212.169 ms/token
```

本次目标是 5 分钟正确性与稳定性，不把该 eager、dummy-weight 数字作为正式
LoongServe 性能结论。

## 资源回收

结束后：

- Ray：`0/192 CPU`、`0/8 GPU`、无 pending demand；
- 8 张 H200：全部 `0 MiB / 0%`；
- 无残留 `ModelRunner` 或 longrun benchmark 进程。

## 产物

- `ls_loop16_fixed_tail_longrun_8gpu_5min_20260723.json`
- `ls_loop16_fixed_tail_longrun_8gpu_5min_20260723.manifest.json`
- `ls_loop16_fixed_tail_longrun_8gpu_5min_20260723.jsonl`
- `ls_loop16_fixed_tail_longrun_8gpu_5min_20260723.log`
