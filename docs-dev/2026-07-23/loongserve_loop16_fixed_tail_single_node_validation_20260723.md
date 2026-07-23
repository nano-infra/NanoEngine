# LoongServe-style `loop_count=16` 固定尾轮单机验证

时间：2026-07-23 03:25--03:29 UTC

## 结论

提交 `54e425a fix: keep LoongServe decode loops fixed` 的修改在当前 HEAD
`4d71131` 上奏效。

单机 8×H200 的 LoongServe-style Decode-only 路径中，配置
`loop_count=16` 后，即使所有请求在尾轮只剩 1 个有效输出 token：

- Scheduler 仍发布 `execution_loop_count=16`；
- 8 个 ModelRunner 均完成 16 次 forward，聚合 CUDA-event 日志记录
  `model_forward_gpu_loop_count: 16`；
- postprocess 只保留达到 `max_tokens` 前的有效 token；
- 8/8 请求全部完成，没有超出可见输出长度。

因此旧行为 `[16, 1]` 已被当前固定执行语义 `[16, 16]` 替代。

这里的 LoongServe-style 指 NanoDeploy 的 DP1×SP8/EP8 Decode scheduler、
placement 和 data plane；`loop_count=16` 是 NanoDeploy 的 chunked Decode
扩展，不等同于原生 LoongServe 每次 manager step 生成一个 token 的 cadence。

## 环境

- Ray：`10.102.206.14:7789`
- 节点：1
- GPU：8×NVIDIA H200
- 测试前：Ray `0/8 GPU`，8 张卡均为 `0 MiB / 0%`
- master endpoint：`10.102.206.14:29906`
- 模型：`/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3`
- topology：Attention DP1×SP8×TP1，FFN EP8
- backend：`hao_basic`
- `dummy_prefill=true`、`dummy_weight=true`
- 当前工作树包含用户原有的 `ray_executor.py` RPC buffer override；本任务没有
  修改它。

当前实际加载的是 2026-07-22 重新构建的 installed C++ extension；它与
当前固定尾轮 CPU 回归行为一致，所以本轮没有重复 editable reinstall。

## CPU 定向回归

运行：

```bash
PYTHONMALLOC=debug pytest -q \
  tests/test_ls_decode_scheduler.py::test_chunked_16_reserves_kv_and_keeps_fixed_final_decode_step \
  tests/test_llm_engine_kv_maintenance.py::test_actual_engine_chunked_16_keeps_fixed_loop_for_one_token_tail \
  tests/test_ls_decode_scheduler.py::test_max_tokens_two_runs_exactly_one_real_decode_after_bootstrap \
  tests/test_llm_engine_kv_maintenance.py::test_actual_scheduler_max_tokens_two_decodes_exactly_once
```

结果：`4 passed in 2.80s`。

## 8-GPU 权威尾轮用例

命令：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  NANODEPLOY_LOG_MODEL_FORWARD_TIMING=1 \
  PYTHONUNBUFFERED=1 \
  python scripts/bench_ls_decode_itl_compare.py \
  --case ls \
  --ray-address 10.102.206.14:7789 \
  --master-address 10.102.206.14:29906 \
  --num-requests 8 \
  --prompt-len 63 \
  --max-tokens 18 \
  --max-num-seqs 64 \
  --loop-count 16 \
  --discard-decode-steps 0 \
  --max-steps 4 \
  --output-json \
    docs-dev/2026-07-23/ls_loop16_fixed_tail_cross_block_single_node_8gpu_20260723.json
```

`prompt_len=63`、KV block size 64，因而首个 16-token chunk 同时覆盖跨 block
reservation。Admission bootstrap 先完成 1 token；第一次 Decode 生成 16 个有效
token 后，每个请求只剩 1 个有效 token。第二次 Decode 是本修复的关键尾轮。

机器断言：

```text
total steps:                 3  (1 admission + 2 decode)
execution_loop_count:       [16, 16]
num_tokens_returned:        [-128, -128]  # 8 requests × 16 slots
finished_requests_by_step:  [0, 0, 8]
outputs_by_step:            [0, 0, 8]
worker timing K=16 records: 2
engine execution K=16:      2
process exit code:          0
```

第二轮日志同时包含：

```text
model_forward_gpu_loop_count: 16
'execution_loop_count': 16
```

日志没有 inconsistent-loop warning、scheduler fatal、traceback、CUDA error 或
NCCL error。Ray metrics exporter 有不影响计算与结果的 observability warning。

## 额外交叉验证

还运行了 `prompt_len=4096`、`max_tokens=34` 的 8-request 用例。它完成
1 次 admission 和 3 次 Decode：

```text
execution_loop_count:       [16, 16, 16]
num_tokens_returned:        [-128, -128, -128]
finished_requests_by_step:  [0, 0, 0, 8]
```

该用例的 Python benchmark 正常完成并写出 JSON。首次启动时当天产物目录尚不存在，
因此外层 `tee` 未能创建日志并使 shell pipeline 返回 1；这不是 benchmark 失败。
上面的权威尾轮用例在目录已存在后完整保存日志并以进程 exit 0 结束。

## 资源回收

测试后：

- Ray：`0/192 CPU`、`0/8 GPU`，无 pending demand；
- 8 张 H200：全部 `0 MiB / 0%`；
- 无残留 `ModelRunner` 或 benchmark 进程。

## 产物

- `ls_loop16_fixed_tail_cross_block_single_node_8gpu_20260723.json`
- `ls_loop16_fixed_tail_cross_block_single_node_8gpu_20260723.log`
- `ls_loop16_fixed_tail_single_node_8gpu_20260723.json`

2026-07-20 的旧文档和 JSON 中 `[16, 16, 1]` 是 `54e425a` 之前的语义，
不能再作为当前实现的预期值。
