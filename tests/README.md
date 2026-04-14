# Tests

## `test_mla_sp_backend_correctness.py`

这个脚本用于对比 NanoDeploy 里的两个 MLA SP all2all 后端是否一致：

- `legacy_ll`
- `hao_basic`

脚本会走仓库里的正式后端切换路径：

`set_sp_context(..., backend="legacy_ll" | "hao_basic")`

覆盖三类 MLA 通信：

- `Q`：masked non-transpose + offsets
- `Res`：masked transpose
- `Lse`：masked transpose

如果使用 `--mode both`，还会额外校验：

- eager 结果正确
- CUDAGraph replay 结果正确
- eager 和 graph 一致
- `legacy_ll` 和 `hao_basic` 一致

默认还会在每次 all2all 前插一个 `all_reduce` 作为同步前导，用来让多 rank 的 launch 更稳定，尤其是 graph 模式下的 8 卡场景。

## 基本用法

先进入仓库根目录：

```bash
cd /mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-April
```

然后用 `torchrun` 启动。这里脚本把 `WORLD_SIZE` 直接当作本次测试的 `SP size`，所以：

- `--nproc_per_node=2` 就是在测 `SP=2`
- `--nproc_per_node=4` 就是在测 `SP=4`
- `--nproc_per_node=8` 就是在测 `SP=8`

最常用命令：

```bash
torchrun --nproc_per_node=8 tests/test_mla_sp_backend_correctness.py --mode both
```

只测 eager：

```bash
torchrun --nproc_per_node=8 tests/test_mla_sp_backend_correctness.py --mode eager
```

只测 graph：

```bash
torchrun --nproc_per_node=8 tests/test_mla_sp_backend_correctness.py --mode graph
```

## 不同 SP Size 怎么跑

### 单独跑某个 SP size

SP=2:

```bash
torchrun --nproc_per_node=2 tests/test_mla_sp_backend_correctness.py --mode both
```

SP=4:

```bash
torchrun --nproc_per_node=4 tests/test_mla_sp_backend_correctness.py --mode both
```

SP=8:

```bash
torchrun --nproc_per_node=8 tests/test_mla_sp_backend_correctness.py --mode both
```

### 连续扫多个 SP size

```bash
for sp in 2 4 8; do
  echo "===== SP=${sp} ====="
  torchrun --nproc_per_node=${sp} tests/test_mla_sp_backend_correctness.py --mode both
done
```

如果只想先做快速 smoke：

```bash
for sp in 2 4 8; do
  echo "===== SP=${sp} ====="
  torchrun --nproc_per_node=${sp} tests/test_mla_sp_backend_correctness.py \
    --mode eager \
    --max-num-seqs 2 \
    --num-requests 2
done
```

## 常用参数

- `--mode {eager,graph,both}`：测试 eager、graph，或者两者都测
- `--reference-backend`：基准后端，默认 `legacy_ll`
- `--candidate-backend`：待验证后端，默认 `hao_basic`
- `--dtype {float16,bfloat16}`：默认 `bfloat16`
- `--max-num-seqs`：SP buffer 的 `max_num_seqs`
- `--num-requests`：本轮激活的请求数，要求 `num_requests <= max_num_seqs`
- `--warmup`：graph capture 前的 warmup 次数
- `--graph-replays`：capture 后 replay 次数
- `--preamble {none,all_reduce,all_gather}`：all2all 前的同步前导，默认 `all_reduce`

例如，缩小规模跑一个 4 卡 smoke：

```bash
torchrun --nproc_per_node=4 tests/test_mla_sp_backend_correctness.py \
  --mode both \
  --max-num-seqs 2 \
  --num-requests 2 \
  --warmup 1 \
  --graph-replays 1
```

## GPU 选择

默认 `torchrun` 会按可见 GPU 分配 local rank。若只想使用部分卡，可以先限制：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 tests/test_mla_sp_backend_correctness.py --mode both
```

## 通过标准

正常通过时，日志里会看到类似输出：

- `[legacy_ll] Q passed ...`
- `[hao_basic] Res passed ...`
- `[compare] Lse graph: legacy_ll == hao_basic`
- `All MLA SP all-to-all backend checks passed.`

如果失败，脚本会直接抛出 mismatch，并打印：

- 哪个 payload/mode 失败
- 哪个 rank 失败
- 第一个不一致的位置
- 对应的 actual / expected 值

## `benchmark_mla_sp_backend.py`

这个脚本用于在 NanoDeploy 仓库内直接对比两种 MLA SP all2all 后端的性能：

- `legacy_ll`
- `hao_basic`

它和上面的 correctness 脚本不同点在于：

- 直接输出 walltime / trace 统计
- 支持一次 `torchrun` 扫多个 `cp_size`
- 支持按 `fan_out` / `uniform` / `fan_in` 扫不同通信场景
- 支持按 `batch_size` 扫曲线
- 按 `Q` / `Res` / `Lse` 分 payload 记录结果
- 输出 `summary.json`、`comparison.csv` 和 profiler trace

默认 DeepSeek V3 MLA 参数：

- `num_heads=128`
- `kv_lora_rank=512`
- `qk_rope_head_dim=64`
- `Q feature dim = 128 * (512 + 64)`
- `Res feature dim = 128 * 512`
- `Lse feature dim = 128`

常用命令：

```bash
torchrun --nproc_per_node=8 tests/benchmark_mla_sp_backend.py \
  --cp-sizes 2,4,8 \
  --patterns fan_out,uniform,fan_in \
  --batch-sizes 1,2,4,8,16,32 \
  --backends legacy_ll,hao_basic \
  --mode graph
```

快速 smoke：

```bash
torchrun --nproc_per_node=4 tests/benchmark_mla_sp_backend.py \
  --cp-sizes 2,4 \
  --patterns fan_out,uniform,fan_in \
  --batch-sizes 1,2,4,8 \
  --payloads Q,Res,Lse \
  --iters 10 \
  --repeats 2 \
  --profile-iters 4 \
  --warmup 2 \
  --graph-inner-iters 8
```

只跑单个 batch size 时也可以继续用旧别名：

```bash
torchrun --nproc_per_node=8 tests/benchmark_mla_sp_backend.py \
  --cp-sizes 8 \
  --patterns fan_out \
  --num-requests 8 \
  --payloads Q,Res,Lse
```

默认输出位置：

- `profiler_traces/mla_sp_backend_bench/summary.json`
- `profiler_traces/mla_sp_backend_bench/comparison.csv`
- `profiler_traces/mla_sp_backend_bench/traces/`

CSV 里会带这些关键字段，便于后续画曲线：

- `cp_size`
- `pattern`
- `batch_size`
- `payload`
- `all2all_mean_us`
- `comm_region_mean_us`
- `local_send_bytes_sum`

## `plot_mla_sp_backend_csv.py`

这个脚本用于把上面的 CSV 直接画成曲线图和 speedup 图：

```bash
python utils_analysis/plot_mla_sp_backend_csv.py \
  profiler_traces/mla_sp_backend_bench/comparison.csv \
  --metric all2all_mean_us \
  --x-log2
```

如果只想看某个 `cp_size` / `payload`：

```bash
python utils_analysis/plot_mla_sp_backend_csv.py \
  profiler_traces/mla_sp_backend_bench/comparison.csv \
  --metric comm_region_mean_us \
  --cp-sizes 8 \
  --payloads Q,Res \
  --x-log2
```

默认会输出：

- `latency_cp*_*.png`
- `speedup_cp*_*.png`
