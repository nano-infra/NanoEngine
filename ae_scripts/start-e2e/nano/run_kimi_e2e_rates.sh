#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_SCRIPT="$SCRIPT_DIR/bench_serving_overhead.py"

# Machine-specific roots come from paths.env, the single place in this tree
# that may hold absolute paths. Existing environment variables win.
if [[ -f "$AE_ROOT/paths.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$AE_ROOT/paths.env"
    set +a
fi
require_env() {
    if [[ -z "${!1:-}" ]]; then
        echo "$1 is not set. Add it to $AE_ROOT/paths.env or export it." >&2
        exit 1
    fi
}

# NanoDeploy is the implementation under test. The launch chain itself stays
# in ae_scripts; no shell/Python script is imported from the NanoDeploy checkout.
NANODEPLOY_WORKDIR="${NANODEPLOY_WORKDIR:-$(cd "$AE_ROOT/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RAY_ADDR="${RAY_ADDR:-10.102.252.174:6380}"
MASTER_ADDR="${MASTER_ADDR:-10.102.252.174:29500}"
MODEL_PATH="${MODEL_PATH:-${AE_KIMI_MODEL:-}}"
DATASET_PATH="${DATASET_PATH:-${AE_DATASET_MIXLONG_0326:-}/sharegpt4o-random_geminiissue_r0.05_n60000_60k.csv}"
require_env MODEL_PATH
require_env DATASET_PATH

REQUEST_RATES="${REQUEST_RATES:-25 30 35}"
SEND_DURATION_SEC="${SEND_DURATION_SEC:-600}"
RATE_COOLDOWN_SEC="${RATE_COOLDOWN_SEC:-15}"
MAX_REQUEST_TOKENS="${MAX_REQUEST_TOKENS:-0}"
MAX_INPUT_LEN="${MAX_INPUT_LEN-1000000}"

# Four nodes x 8 H200 GPUs: four DP groups, each with eight SP ranks.
DP_SIZE="${DP_SIZE:-4}"
SP_SIZE="${SP_SIZE:-8}"
TP_SIZE="${TP_SIZE:-1}"
EP_SIZE=$((DP_SIZE * SP_SIZE * TP_SIZE))
BATCH_SIZE="${BATCH_SIZE:-256}"
GPU_MEM_GB="${GPU_MEM_GB:-141}"
GPU_UTIL="${GPU_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
LOOP_COUNT="${LOOP_COUNT:-16}"
SP_BACKEND="${SP_BACKEND:-hao_basic}"
CUDA_GRAPH_MODE="${CUDA_GRAPH_MODE:-full}"
ROUTING_STRATEGY="${ROUTING_STRATEGY:-LeastBatch}"

# These two values are intentionally fixed for the Nano paper run:
#   legacy_global = centralized scheduler architecture
#   bucket        = explicit sequence-length buckets choose SP/CP size
# This is neither long_short_sp8 nor the legacy segment-size search.
SCHEDULER_ARCH="legacy_global"
DYNAMIC_SP_SIZE_STRATEGY="bucket"
DYNAMIC_SP_BUCKET_PRESET="${DYNAMIC_SP_BUCKET_PRESET:-kimi_k2}"
DYNAMIC_SP_BUCKET_POLICY="${DYNAMIC_SP_BUCKET_POLICY:-}"
RESOLVED_KIMI_K2_BUCKET_POLICY="1:1024-10240;2:10241-22528;3:22529-190464;4:190465-210944;5:210945-354304;6:354305-624640;7:624641-673792;8:673793-1000000"

# segment_size remains an internal KV-accounting granularity. In bucket mode it
# does not decide the participating SP size.
KV_ACCOUNTING_SEGMENT_SIZE="${KV_ACCOUNTING_SEGMENT_SIZE:-65536}"

RUN_TAG="${RUN_TAG:-kimi_issue005_rates}"
E2E_LOG_ROOT="${E2E_LOG_ROOT:-$AE_ROOT/bench_logs/e2e}"
OUTPUT_DIR="${OUTPUT_DIR:-$E2E_LOG_ROOT/$RUN_TAG}"
DRY_RUN="${DRY_RUN:-0}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

log() {
    local message="$*"
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$message" \
        | tee -a "$PROGRESS_LOG"
}

rate_to_num_requests() {
    local rate="$1"
    awk -v rate="$rate" -v duration="$SEND_DURATION_SEC" \
        'BEGIN {printf "%d", int(rate * duration + 0.5)}'
}

[[ -f "$BENCH_SCRIPT" ]] || die "missing local benchmark driver: $BENCH_SCRIPT"
[[ -d "$NANODEPLOY_WORKDIR/nanodeploy" ]] \
    || die "invalid NANODEPLOY_WORKDIR: $NANODEPLOY_WORKDIR"
grep -Fq 'KIMI_K2_BUCKET_POLICY' "$NANODEPLOY_WORKDIR/nanodeploy/config.py" \
    || die "NanoDeploy checkout does not support the kimi_k2 bucket preset"
[[ -d "$MODEL_PATH" ]] || die "missing model directory: $MODEL_PATH"
[[ -f "$DATASET_PATH" ]] || die "missing dataset CSV: $DATASET_PATH"

if [[ "$DP_SIZE" -ne 4 || "$SP_SIZE" -ne 8 || "$TP_SIZE" -ne 1 ]]; then
    die "this launcher requires DP_SIZE=4, SP_SIZE=8, TP_SIZE=1; got $DP_SIZE/$SP_SIZE/$TP_SIZE"
fi
[[ "$MAX_REQUEST_TOKENS" =~ ^[0-9]+$ ]] \
    || die "MAX_REQUEST_TOKENS must be a non-negative integer"
if [[ -n "$MAX_INPUT_LEN" && ! "$MAX_INPUT_LEN" =~ ^[1-9][0-9]*$ ]]; then
    die "MAX_INPUT_LEN must be empty or a positive integer"
fi
awk -v duration="$SEND_DURATION_SEC" 'BEGIN {exit !(duration > 0)}' \
    || die "SEND_DURATION_SEC must be positive"
awk -v cooldown="$RATE_COOLDOWN_SEC" 'BEGIN {exit !(cooldown >= 0)}' \
    || die "RATE_COOLDOWN_SEC must be non-negative"

if [[ -n "$DYNAMIC_SP_BUCKET_POLICY" ]]; then
    [[ "$DYNAMIC_SP_BUCKET_PRESET" == "none" ]] \
        || die "set DYNAMIC_SP_BUCKET_PRESET=none when using DYNAMIC_SP_BUCKET_POLICY"
else
    [[ "$DYNAMIC_SP_BUCKET_PRESET" == "kimi_k2" ]] \
        || die "Kimi bucket mode requires preset=kimi_k2 or an explicit policy"
fi

read -r -a rate_list <<< "$REQUEST_RATES"
[[ ${#rate_list[@]} -gt 0 ]] || die "REQUEST_RATES must contain at least one rate"
for rate in "${rate_list[@]}"; do
    awk -v rate="$rate" 'BEGIN {exit !(rate > 0)}' \
        || die "every request rate must be positive; got $rate"
done

[[ ! -e "$OUTPUT_DIR" ]] || die "refusing to overwrite output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

PROGRESS_LOG="$OUTPUT_DIR/run.progress"
SUMMARY_FILE="$OUTPUT_DIR/run_summary.tsv"
CONFIG_FILE="$OUTPUT_DIR/config.txt"

printf 'rate\tnum_requests\tstatus\tlog_dir\n' > "$SUMMARY_FILE"
{
    printf 'RUN_TAG=%q\n' "$RUN_TAG"
    printf 'NANODEPLOY_WORKDIR=%q\n' "$NANODEPLOY_WORKDIR"
    printf 'RAY_ADDR=%q\n' "$RAY_ADDR"
    printf 'MASTER_ADDR=%q\n' "$MASTER_ADDR"
    printf 'MODEL_PATH=%q\n' "$MODEL_PATH"
    printf 'DATASET_PATH=%q\n' "$DATASET_PATH"
    printf 'REQUEST_RATES=%q\n' "$REQUEST_RATES"
    printf 'SEND_DURATION_SEC=%q\n' "$SEND_DURATION_SEC"
    printf 'MAX_REQUEST_TOKENS=%q\n' "$MAX_REQUEST_TOKENS"
    printf 'MAX_INPUT_LEN=%q\n' "$MAX_INPUT_LEN"
    printf 'TOPOLOGY=dp%q_sp%q_tp%q_ep%q\n' "$DP_SIZE" "$SP_SIZE" "$TP_SIZE" "$EP_SIZE"
    printf 'SCHEDULER_ARCH=%q\n' "$SCHEDULER_ARCH"
    printf 'DYNAMIC_SP_SIZE_STRATEGY=%q\n' "$DYNAMIC_SP_SIZE_STRATEGY"
    printf 'DYNAMIC_SP_BUCKET_PRESET=%q\n' "$DYNAMIC_SP_BUCKET_PRESET"
    printf 'DYNAMIC_SP_BUCKET_POLICY=%q\n' "${DYNAMIC_SP_BUCKET_POLICY:-$RESOLVED_KIMI_K2_BUCKET_POLICY}"
    printf 'KV_ACCOUNTING_SEGMENT_SIZE=%q\n' "$KV_ACCOUNTING_SEGMENT_SIZE"
    printf 'ROUTING_STRATEGY=%q\n' "$ROUTING_STRATEGY"
    printf 'SP_BACKEND=%q\n' "$SP_BACKEND"
    printf 'CUDA_GRAPH_MODE=%q\n' "$CUDA_GRAPH_MODE"
    printf 'OUTPUT_DIR=%q\n' "$OUTPUT_DIR"
} > "$CONFIG_FILE"

# Ray/GCS traffic and the benchmark driver must bypass HTTP proxies.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# TCP/bootstrap network.
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"

# NCCL and DeepEP/NVSHMEM RDMA settings used by the four-node environment.
export NCCL_IB_HCA="${NCCL_IB_HCA:-=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_TC="${NCCL_IB_TC:-186}"
export SLIME_VISIBLE_DEVICES="${SLIME_VISIBLE_DEVICES:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export SLIME_GID_INDEX="${SLIME_GID_INDEX:-3}"
export SLIME_QP_NUM="${SLIME_QP_NUM:-4}"
export DEEPEP_SMS="${DEEPEP_SMS:-16}"
export DEEPEP_MAX_TOKENS_PER_RANK="${DEEPEP_MAX_TOKENS_PER_RANK:-256}"
export DEEPEP_ENABLE_MNNVL="${DEEPEP_ENABLE_MNNVL:-0}"
export DEEPEP_MODE="${DEEPEP_MODE:-auto}"
export NVSHMEM_QP_DEPTH="${NVSHMEM_QP_DEPTH:-1024}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NANODEPLOY_LOG_DECODE_STEP_DETAIL="${NANODEPLOY_LOG_DECODE_STEP_DETAIL:-1}"
export RAY_DEDUP_LOGS=0

build_lib="$NANODEPLOY_WORKDIR/build/lib"
if [[ -d "$build_lib" ]]; then
    export PYTHONPATH="$NANODEPLOY_WORKDIR:$build_lib${PYTHONPATH:+:$PYTHONPATH}"
else
    export PYTHONPATH="$NANODEPLOY_WORKDIR${PYTHONPATH:+:$PYTHONPATH}"
fi
export NANODEPLOY_WORKDIR

unset DG_PRINT_CONFIGS DG_JIT_DEBUG
unset NANODEPLOY_MOE_GEMM_DEBUG NANODEPLOY_MOE_GEMM_DEBUG_RANKS
unset NANODEPLOY_MOE_GEMM_DEBUG_LAYERS NANODEPLOY_MOE_GEMM_DEBUG_GEMMS
unset NANODEPLOY_MOE_GEMM_DEBUG_MAX_CALLS NANODEPLOY_MOE_GEMM_DEBUG_SAMPLE_ELEMENTS

log "START run_tag=$RUN_TAG rates=$REQUEST_RATES duration=${SEND_DURATION_SEC}s"
log "scheduler=$SCHEDULER_ARCH dynamic_sp=$DYNAMIC_SP_SIZE_STRATEGY bucket_preset=$DYNAMIC_SP_BUCKET_PRESET"
log "dataset=$DATASET_PATH output_dir=$OUTPUT_DIR"

last_rate_index=$((${#rate_list[@]} - 1))
for rate_index in "${!rate_list[@]}"; do
    rate="${rate_list[$rate_index]}"
    num_requests="$(rate_to_num_requests "$rate")"
    rate_tag="${rate//./p}"
    rate_log_dir="$OUTPUT_DIR/rate_${rate_tag}"
    driver_log="$rate_log_dir/driver.log"
    itl_log="$rate_log_dir/itl_samples.jsonl"
    command_file="$rate_log_dir/command.txt"
    mkdir -p "$rate_log_dir"

    cmd=(
        "$PYTHON_BIN" -u "$BENCH_SCRIPT"
        --dataset csv
        --csv-path "$DATASET_PATH"
        --max-request-tokens "$MAX_REQUEST_TOKENS"
        --num-requests "$num_requests"
        --request-rate "$rate"
        --sp "$SP_SIZE"
        --dp "$DP_SIZE"
        --ep "$EP_SIZE"
        --tp "$TP_SIZE"
        --max-num-seqs "$BATCH_SIZE"
        --gpu-memory-limit-gb "$GPU_MEM_GB"
        --gpu-memory-utilization "$GPU_UTIL"
        --max-model-len "$MAX_MODEL_LEN"
        --dummy-prefill
        --ray-address "$RAY_ADDR"
        --master-address "$MASTER_ADDR"
        --loop-count "$LOOP_COUNT"
        --model-path "$MODEL_PATH"
        --routing-strategy "$ROUTING_STRATEGY"
        --itl-log-path "$itl_log"
        --segment-size "$KV_ACCOUNTING_SEGMENT_SIZE"
        --sp-backend "$SP_BACKEND"
        --cuda-graph-mode "$CUDA_GRAPH_MODE"
        --scheduler-arch "$SCHEDULER_ARCH"
        --fixed-sp-size 0
        --enable-dynamic-sp-size
        --dynamic-sp-size-strategy "$DYNAMIC_SP_SIZE_STRATEGY"
    )
    if [[ -n "$MAX_INPUT_LEN" ]]; then
        cmd+=(--max-input-len "$MAX_INPUT_LEN")
    fi
    if [[ -n "$DYNAMIC_SP_BUCKET_POLICY" ]]; then
        cmd+=(--dynamic-sp-bucket-policy "$DYNAMIC_SP_BUCKET_POLICY")
    else
        cmd+=(--dynamic-sp-bucket-preset "$DYNAMIC_SP_BUCKET_PRESET")
    fi

    {
        printf 'cd %q\n' "$NANODEPLOY_WORKDIR"
        printf 'NANODEPLOY_WORKDIR=%q RAY_DEDUP_LOGS=0 ' "$NANODEPLOY_WORKDIR"
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } > "$command_file"

    log "RUN rate=$rate num_requests=$num_requests log_dir=$rate_log_dir"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY_RUN: '
        tail -n 1 "$command_file"
        printf '%s\t%s\tplanned\t%s\n' \
            "$rate" "$num_requests" "$rate_log_dir" >> "$SUMMARY_FILE"
        continue
    fi

    set +e
    (
        cd "$NANODEPLOY_WORKDIR"
        "${cmd[@]}"
    ) 2>&1 | tee "$driver_log"
    run_status=${PIPESTATUS[0]}
    set -e

    if [[ "$run_status" -ne 0 ]]; then
        printf '%s\t%s\tfailed\t%s\n' \
            "$rate" "$num_requests" "$rate_log_dir" >> "$SUMMARY_FILE"
        log "FAILED rate=$rate status=$run_status; stopping sweep"
        exit "$run_status"
    fi

    printf '%s\t%s\tok\t%s\n' \
        "$rate" "$num_requests" "$rate_log_dir" >> "$SUMMARY_FILE"
    log "DONE rate=$rate num_requests=$num_requests"

    if [[ "$rate_index" -lt "$last_rate_index" ]] \
        && awk -v cooldown="$RATE_COOLDOWN_SEC" 'BEGIN {exit !(cooldown > 0)}'; then
        log "COOLDOWN ${RATE_COOLDOWN_SEC}s before next rate"
        sleep "$RATE_COOLDOWN_SEC"
    fi
done

if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN complete; no benchmark was launched"
else
    log "COMPLETE all rates finished and all submitted requests drained"
fi
