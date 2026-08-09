# DeepSeek DP8/EP8 Native Backend Smoke

Date: 2026-08-09 UTC

## Objective

Validate the post-dlBLAS-removal DeepSeek-V3 decode path on one eight-GPU
node using the Ray cluster at `10.102.252.174:6380`. The target path is native
DeepEP low-latency dispatch/combine with native DeepGEMM masked grouped FP8
gate-up and down GEMMs.

## Environment

- Git commit: `386c04e` (`build: remove dlblas dependency`)
- Model config: `/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3`
- Topology: DP8/SP1/EP8, eager, one Ray node with eight H200 GPUs
- DeepGEMM: `2.3.0+477618c`
- DeepEP: `1.2.1+73b6ea4`
- `SLIME_QP_NUM=4`
- HTTP(S) and `ALL_PROXY` variables unset for Ray
- Effective DeepEP configuration on every rank:
  `DEEPEP_SMS=16`, `DEEPEP_MAX_TOKENS_PER_RANK=8`,
  `DEEPEP_ENABLE_MNNVL=0`, `DEEPEP_MODE=auto`,
  `NVSHMEM_QP_DEPTH=1024`, `num_qps_per_rank=32`

## Command

The retry used the following test arguments after exporting the environment
listed above and enabling rank-0 layer-3 gate-up/down debug records:

```bash
python -u examples/dummy_prefill.py \
  --num-seqs 8 \
  --seq-len 64 \
  --max-tokens 2 \
  --max-num-seqs 8 \
  --dp 8 \
  --sp 1 \
  --ep 8 \
  --model-path /mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3 \
  --master-address 10.102.252.174:26444 \
  --ray-address 10.102.252.174:6380 \
  --max-num-send-seqs 8 \
  --max-num-recv-seqs 10 \
  --gpu-memory-utilization 0.9 \
  --loop-count 1 \
  --max-model-len 1024 \
  --max-num-batched-tokens 1024 \
  --routing-strategy LeastBatch \
  --enforce-eager
```

Debug environment:

```bash
export DG_PRINT_CONFIGS=1
export NANODEPLOY_MOE_GEMM_DEBUG=1
export NANODEPLOY_MOE_GEMM_DEBUG_RANKS=0
export NANODEPLOY_MOE_GEMM_DEBUG_LAYERS=3
export NANODEPLOY_MOE_GEMM_DEBUG_GEMMS=gate_up,down
export NANODEPLOY_MOE_GEMM_DEBUG_MAX_CALLS=1
export NANODEPLOY_MOE_GEMM_DEBUG_SAMPLE_ELEMENTS=0
```

## Result

Passed with exit code 0.

- All eight ranks passed the exact backend version/symbol and collective
  configuration checks.
- All eight ranks initialized 6,957 KV-cache blocks.
- The layer-3 debug record was emitted after native DeepEP dispatch and before
  each native DeepGEMM call. Both masked grouped GEMMs executed:
  - gate-up: local expert tensor shape `[32, 64, 7168]`, `masked_m_sum=7`;
  - down: local expert tensor shape `[32, 64, 2048]`, `masked_m_sum=7`.
- The end-to-end request run completed 8/8 requests and produced 16 decode
  tokens from 512 prompt tokens.
- Every rank explicitly destroyed its DeepEP buffer. All eight worker actors
  terminated successfully, and the Ray cluster returned to `0.0/8.0 GPU` use.
- No Python traceback, CUDA OOM, actor death, backend contract error, or DeepEP
  destroy warning occurred in the successful retry.

Ray metrics-exporter connection warnings were present but did not affect model
initialization, collectives, decode, or cleanup.

## Initial Attempt

The first attempt at 04:50 UTC stopped during model construction because eight
Ray-external processes already occupied approximately 133 GiB on every GPU.
It did not reach model forward and therefore was not used as backend validation.
After those workloads released the GPUs, the identical smoke configuration
passed on retry at 06:28 UTC.

## Validation Boundary

This run validates the single-node eager decode path with dummy weights and real
DeepEP/DeepGEMM CUDA execution. It does not cover full or piecewise CUDA Graph,
larger decode batches, real-weight numerical correctness, or multi-node EP.
