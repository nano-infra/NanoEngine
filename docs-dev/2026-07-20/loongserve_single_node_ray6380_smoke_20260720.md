# LoongServe-style 单机 Ray 6380 serving smoke

日期：2026-07-20

## 结论

当前工作树的 LoongServe-style Decode-only 路径可以完成单机真实 serving：在
`10.102.252.174:6380` 的单节点 Ray 集群上，Attention `DP1 × SP8` / FFN `EP8`
使用 8 张 H200，以 eager 模式运行 Issue-1% CSV 前 8 条请求，最终 `8 / 8` 全部完成，
进程 exit code 为 0，manifest 状态为 `success`。运行期间没有 scheduler fatal、planner
failure、OOM、CUDA/NCCL error 或 Ray worker crash。

这是一项功能 smoke，不是性能验收。8 个请求都能经过 admission、数百轮真实 Decode、
finish/release 并落盘；但该小 batch 没有覆盖高压力下的 OFFLOAD、memory scale-up 或 KV
consolidation，且 eager 结果不能替代 full CUDA Graph 测试。

## 环境与配置

- 当前 HEAD：`738b5ee27f27e5bd944e4550a72223112fbadbab`；LoongServe-style 主实现提交为
  `b8e0b7d`。
- 当前工作树包含用户既有的 RPC endpoint buffer override：
  `8 * 32_000_000 -> 32 * 32_000_000`；本次未修改该文件。
- Ray：`10.102.252.174:6380`，1 个 active node，192 CPU、8 H200；运行前
  `0 / 8 GPU` 被占用。
- NCCL/master：`10.102.252.174:29906`。
- 模型：本地 DeepSeek-V3，`dummy_prefill=true`、`dummy_weight=true`。
- topology：Attention `DP1 × SP8 × TP1`，FFN `DP1 × EP8 × TP1`。
- profile：`loong_decode_issue001`，`ls_max_num_ooe=8`，future-KV admission、
  memory scale-up 和 execute-mode KV consolidation 均启用。
- workload：Issue-1% CSV 前 8 条，seed 0，4 req/s；prompt 共 1,588 tokens，
  output 共 5,459 tokens，单请求 output length 为
  `[698, 801, 529, 723, 690, 436, 865, 717]`。
- Ray 连接与 NanoDeploy 运行前均清除了四个 HTTP(S) proxy 环境变量。

## 命令

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  PYTHONUNBUFFERED=1 \
  python scripts/bench_ls_decode_serving.py \
  --ray-address 10.102.252.174:6380 \
  --master-address 10.102.252.174:29906 \
  --attention-dp 1 \
  --attention-sp 8 \
  --duration-sec 2 \
  --request-rate 4 \
  --num-requests 8 \
  --enforce-eager \
  --ls-max-num-ooe 8 \
  --output-jsonl docs-dev/2026-07-20/ls_decode_single_node_serving_smoke_8gpu_ray6380_20260720.jsonl \
  --manifest-json docs-dev/2026-07-20/ls_decode_single_node_serving_smoke_8gpu_ray6380_20260720.manifest.json
```

## 结果

- wall time：237.88 s；requests：`8 sent / 8 completed`；
- throughput：22.95 output token/s；
- average TTFT：0.26 ms；average E2E：185.01 s；
- TPOT without queue：avg 270.02 ms/token，P99 294.91 ms/token；
- ITL with decode queue：avg 270.06 ms/token，P99 274.40 ms/token；
- queueing：avg 0.26 ms，P99 0.34 ms；
- 100 ms TPOT SLO：`0 / 8`，goodput 0%。

因此当前结果只支持“功能路径可运行”，不支持“性能已经达标”。

## 退出后的资源检查

测试进程 exit 0 后，Ray 报告 `0 / 192 CPU`、`0 / 8 GPU`、无 pending demand；GCS
actor table 也为 `ALIVE_ACTORS=[]`。Ray rank0 worker 日志显示 actor 收到预期的
`ray.kill` 并退出。

但 `nvidia-smi` 随后持续显示 GPU0 仍有约 123.3--123.7 GiB 的 context；中间两次
utilization 为 100%，最终快照已降为 0%，但显存仍未释放。process name 为 `[Not Found]`，
当前 PID namespace 中也找不到对应进程；其余 7 张 GPU 均为 0 MiB。由于 Ray/GCS 已无
活跃 actor，且 GPU reset 会影响共享节点，本次没有擅自执行 `nvidia-smi --gpu-reset` 或停止
Ray。后续再次占用该节点前应先由节点所有者确认并清理这个 driver/container 层的残留 context。

## 产物

- `ls_decode_single_node_serving_smoke_8gpu_ray6380_20260720.manifest.json`
- `ls_decode_single_node_serving_smoke_8gpu_ray6380_20260720.jsonl`
- 本报告
