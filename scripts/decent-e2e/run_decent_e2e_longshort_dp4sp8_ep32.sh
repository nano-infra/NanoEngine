#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
START_BENCH_SH="$ROOT_DIR/scripts/sp_ablation/start_bench.sh"
BENCHMARK_PY="$ROOT_DIR/scripts/sp_ablation/bench_serving_overhead.py"
CONFIG_PY="$ROOT_DIR/nanodeploy/config.py"
VALIDATE_RUN_PY="$SCRIPT_DIR/validate_run.py"

# Cluster addresses. Keep the old experiment's rendezvous port while moving
# both Ray and the torch distributed master to the new head-node IP.
RAY_ADDR="${RAY_ADDR:-10.102.234.33:7789}"
MASTER_ADDR="${MASTER_ADDR:-10.102.234.33:27799}"

MODEL_PATH="${MODEL_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3}"
ISSUE001_DATASET="${ISSUE001_DATASET:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv}"
ISSUE005_DATASET="${ISSUE005_DATASET:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.05_n60000_60k.csv}"

# The workload envelope intentionally matches the April centralized E2E run.
DURATION_SECONDS="${DURATION_SECONDS:-600}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEGMENT_SIZE="${SEGMENT_SIZE:-65536}"
GPU_MEMORY_LIMIT_GB="${GPU_MEMORY_LIMIT_GB:-141}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-1000000}"
MAX_REQUEST_TOKENS="${MAX_REQUEST_TOKENS:-1000000}"
LOOP_COUNT="${LOOP_COUNT:-16}"

# Fixed topology: attention DP4 x SP8 x TP1 = world size / FFN EP32.
DP_SIZE=4
SP_SIZE=8
TP_SIZE=1
EP_SIZE=$((DP_SIZE * SP_SIZE * TP_SIZE))

# Current benchmark constants that are kept equal to the old E2E run.
MAX_NUM_BATCHED_TOKENS=1024000
KV_CACHE_BLOCK_SIZE=64
MAX_NUM_RECV_SEQS=32
MAX_NUM_SEND_SEQS_KEYWORD=16
WARMUP_REQUESTS=256

# Long-short and decentralized scheduling policy.
LONG_REQUEST_SP_THRESHOLD="${LONG_REQUEST_SP_THRESHOLD:-100000}"
LONG_REQUEST_SP_SIZE="${LONG_REQUEST_SP_SIZE:-8}"
ROUTING_STRATEGY="${ROUTING_STRATEGY:-LeastBatch}"
ROUTER_POLICY="${ROUTER_POLICY:-least_batch_v2}"
SP_MASTER_SELECTOR="${SP_MASTER_SELECTOR:-LeastBatch}"
SP_BACKEND="${SP_BACKEND:-hao_basic}"
CUDA_GRAPH_MODE="${CUDA_GRAPH_MODE:-full}"
DIAGNOSTIC_LOG_INTERVAL="${DIAGNOSTIC_LOG_INTERVAL:-1}"
SLOW_ADD_THRESHOLD_MS="${SLOW_ADD_THRESHOLD_MS:-20}"

ISSUE001_RATES="${ISSUE001_RATES:-10 20 30 35 40 50 60 70 80 90}"
ISSUE005_RATES="${ISSUE005_RATES:-5 10 20 30 40 45}"
read -r -a ISSUE001_RATE_LIST <<< "$ISSUE001_RATES"
read -r -a ISSUE005_RATE_LIST <<< "$ISSUE005_RATES"

RUN_TAG="${RUN_TAG:-decent_e2e_longshort_dp4sp8_ep32_bs256_$(date -u +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/bench_logs/decent-e2e}"
RUN_DIR="${RUN_DIR:-$LOG_ROOT/$RUN_TAG}"
RUN_LABEL="${RUN_LABEL:-decent_e2e_longshort_ep32}"

DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
MAX_RETRIES="${MAX_RETRIES:-5}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-10}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-20}"

usage() {
    cat <<'EOF'
Usage: run_decent_e2e_longshort_dp4sp8_ep32.sh [issue001] [issue005]

No dataset argument runs issue001 followed by issue005. Environment overrides:
  RUN_TAG=...                       Stable tag; reuse it to resume a run.
  DRY_RUN=1                         Print commands without connecting to Ray.
  ISSUE001_RATES="10 20 ..."        Override the issue-1% rate list.
  ISSUE005_RATES="5 10 ..."         Override the issue-5% rate list.
  DURATION_SECONDS=600              Arrival-window duration per rate.
  FORCE_RERUN=1                     Ignore successful stage markers.
  CONTINUE_ON_ERROR=1               Continue after a stage exhausts retries.
  NANODEPLOY_HIER_WORKER_TRANSPORT=ray  Use Ray worker control instead of ZMQ.
EOF
}

die() {
    echo "Error: $*" >&2
    exit 2
}

log() {
    local now
    now="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "[$now] $*" | tee -a "$PROGRESS_LOG"
}

validate_positive_integer() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer; got '$value'."
}

validate_nonnegative_number() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "$name must be non-negative; got '$value'."
}

validate_toggle() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[01]$ ]] || die "$name must be 0 or 1; got '$value'."
}

validate_rates() {
    local dataset_key="$1"
    shift
    (( $# > 0 )) || die "$dataset_key rate list must not be empty."
    local rate
    for rate in "$@"; do
        validate_positive_integer "$dataset_key rate" "$rate"
    done
}

require_source_line() {
    local file="$1"
    local expected="$2"
    grep -Fq "$expected" "$file" || die "Expected invariant '$expected' was not found in $file. Re-audit the benchmark before running."
}

verify_source_invariants() {
    [[ -f "$START_BENCH_SH" ]] || die "Missing start script: $START_BENCH_SH"
    [[ -f "$BENCHMARK_PY" ]] || die "Missing benchmark: $BENCHMARK_PY"
    [[ -f "$CONFIG_PY" ]] || die "Missing config: $CONFIG_PY"
    [[ -f "$VALIDATE_RUN_PY" ]] || die "Missing result validator: $VALIDATE_RUN_PY"

    require_source_line "$BENCHMARK_PY" "max_num_batched_tokens=1024000"
    require_source_line "$BENCHMARK_PY" "kvcache_block_size=64"
    require_source_line "$BENCHMARK_PY" "max_num_recv_seqs=32"
    require_source_line "$BENCHMARK_PY" "max_num_send_seqs=16"
    require_source_line "$BENCHMARK_PY" "dummy_weight=True"
    require_source_line "$BENCHMARK_PY" "perfect_eplb=True"
    require_source_line "$BENCHMARK_PY" "enable_non_uniform_split=not args.disable_non_uniform_split"
    require_source_line "$BENCHMARK_PY" "num_warmup_requests = 256"

    require_source_line "$CONFIG_PY" "load_report_interval_ms: int = 100"
    require_source_line "$CONFIG_PY" "hierarchical_queue_capacity: int = 4096"
    require_source_line "$CONFIG_PY" "max_ingress_batch_requests: int = 256"
    require_source_line "$CONFIG_PY" "max_ingress_drain_ms: float = 0.0"
    require_source_line "$CONFIG_PY" "reserved_blocks_per_req: float = 1.0"
}

write_effective_config() {
    printf "key\tvalue\n"
    printf "ray_addr\t%s\n" "$RAY_ADDR"
    printf "master_addr\t%s\n" "$MASTER_ADDR"
    printf "model_path\t%s\n" "$MODEL_PATH"
    printf "issue001_dataset\t%s\n" "$ISSUE001_DATASET"
    printf "issue005_dataset\t%s\n" "$ISSUE005_DATASET"
    printf "issue001_rates\t%s\n" "$ISSUE001_RATES"
    printf "issue005_rates\t%s\n" "$ISSUE005_RATES"
    printf "duration_seconds\t%s\n" "$DURATION_SECONDS"
    printf "dp\t%s\n" "$DP_SIZE"
    printf "sp\t%s\n" "$SP_SIZE"
    printf "tp\t%s\n" "$TP_SIZE"
    printf "ep\t%s\n" "$EP_SIZE"
    printf "batch_size\t%s\n" "$BATCH_SIZE"
    printf "segment_size\t%s\n" "$SEGMENT_SIZE"
    printf "max_model_len\t%s\n" "$MAX_MODEL_LEN"
    printf "max_input_len\t%s\n" "$MAX_INPUT_LEN"
    printf "max_request_tokens\t%s\n" "$MAX_REQUEST_TOKENS"
    printf "max_num_batched_tokens\t%s\n" "$MAX_NUM_BATCHED_TOKENS"
    printf "kv_cache_block_size\t%s\n" "$KV_CACHE_BLOCK_SIZE"
    printf "max_num_recv_seqs\t%s\n" "$MAX_NUM_RECV_SEQS"
    printf "max_num_send_seqs_keyword\t%s (historical no-op; not a Config field)\n" "$MAX_NUM_SEND_SEQS_KEYWORD"
    printf "warmup_requests\t%s\n" "$WARMUP_REQUESTS"
    printf "gpu_memory_limit_gb\t%s\n" "$GPU_MEMORY_LIMIT_GB"
    printf "gpu_memory_utilization\t%s\n" "$GPU_MEMORY_UTILIZATION"
    printf "loop_count\t%s\n" "$LOOP_COUNT"
    printf "scheduler_arch\thierarchical\n"
    printf "routing_strategy\t%s\n" "$ROUTING_STRATEGY"
    printf "router_policy\t%s\n" "$ROUTER_POLICY"
    printf "sp_master_selector\t%s\n" "$SP_MASTER_SELECTOR"
    printf "dynamic_sp_size\tenabled\n"
    printf "dynamic_sp_size_strategy\tlong_short_sp8\n"
    printf "long_request_sp_threshold\t%s\n" "$LONG_REQUEST_SP_THRESHOLD"
    printf "long_request_sp_size\t%s\n" "$LONG_REQUEST_SP_SIZE"
    printf "sp_backend\t%s\n" "$SP_BACKEND"
    printf "cuda_graph_mode\t%s\n" "$CUDA_GRAPH_MODE"
    printf "worker_transport\t%s\n" "$NANODEPLOY_HIER_WORKER_TRANSPORT"
    printf "slime_qp_num\t%s\n" "$SLIME_QP_NUM"
    printf "deepep_sms\t%s\n" "$DEEPEP_SMS"
    printf "deepep_max_tokens_per_rank\t%s\n" "$DEEPEP_MAX_TOKENS_PER_RANK"
    printf "deepep_enable_mnnvl\t%s\n" "$DEEPEP_ENABLE_MNNVL"
    printf "deepep_mode\t%s\n" "$DEEPEP_MODE"
    printf "diagnostic_log_interval\t%s\n" "$DIAGNOSTIC_LOG_INTERVAL"
    printf "slow_add_threshold_ms\t%s\n" "$SLOW_ADD_THRESHOLD_MS"
}

record_summary() {
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$1" "$2" "$DURATION_SECONDS" "$3" "$4" "$5" "$6" "$7" "$8" "$9" \
        >> "$SUMMARY_TSV"
}

print_command() {
    local stage_log_dir="$1"
    shift
    printf "SLIME_QP_NUM=%q DEEPEP_SMS=%q DEEPEP_MAX_TOKENS_PER_RANK=%q DEEPEP_ENABLE_MNNVL=%q DEEPEP_MODE=%q NANODEPLOY_HIER_WORKER_TRANSPORT=%q BASE_LOG_DIR=%q " \
        "$SLIME_QP_NUM" "$DEEPEP_SMS" "$DEEPEP_MAX_TOKENS_PER_RANK" \
        "$DEEPEP_ENABLE_MNNVL" "$DEEPEP_MODE" \
        "$NANODEPLOY_HIER_WORKER_TRANSPORT" "$stage_log_dir"
    printf "%q " "$@"
    printf "\n"
}

run_one() {
    local dataset_key="$1"
    local dataset_path="$2"
    local rate="$3"
    local num_requests=$((rate * DURATION_SECONDS))
    local stage_log_dir="$RUN_DIR/$dataset_key/rate_${rate}"
    local console_log="$stage_log_dir/console.log"
    local success_marker="$stage_log_dir/SUCCESS"
    local started_at
    local finished_at
    local attempt
    local rc=1
    local validation_output=""

    RUN_INDEX=$((RUN_INDEX + 1))
    mkdir -p "$stage_log_dir"

    if (( RESUME != 0 && FORCE_RERUN == 0 )) && [[ -f "$success_marker" ]]; then
        log "SKIP [$RUN_INDEX/$TOTAL_RUNS] dataset=$dataset_key rate=$rate reason=SUCCESS_marker"
        started_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        record_summary "$dataset_key" "$rate" "$num_requests" "skipped_success" "0" "0" "$started_at" "$started_at" "$stage_log_dir"
        return 0
    fi

    local cmd=(
        bash "$START_BENCH_SH"
        --master-addr "$MASTER_ADDR"
        --ray-addr "$RAY_ADDR"
        --dataset-path "$dataset_path"
        --model-path "$MODEL_PATH"
        --segment-size "$SEGMENT_SIZE"
        --dp-size "$DP_SIZE"
        --sp-size "$SP_SIZE"
        --tp-size "$TP_SIZE"
        --block-size "$KV_CACHE_BLOCK_SIZE"
        --batch-size "$BATCH_SIZE"
        --num-requests "$num_requests"
        --gpu-mem "$GPU_MEMORY_LIMIT_GB"
        --gpu-util "$GPU_MEMORY_UTILIZATION"
        --max-model-len "$MAX_MODEL_LEN"
        --max-input-len "$MAX_INPUT_LEN"
        --max-request-tokens "$MAX_REQUEST_TOKENS"
        --routing-strategy "$ROUTING_STRATEGY"
        --scheduler-arch hierarchical
        --router-policy "$ROUTER_POLICY"
        --sp-master-selector "$SP_MASTER_SELECTOR"
        --loop-count "$LOOP_COUNT"
        --fixed-sp-size 0
        --sp-backend "$SP_BACKEND"
        --cuda-graph-mode "$CUDA_GRAPH_MODE"
        --enable-dynamic-sp-size
        --dynamic-sp-size-strategy long_short_sp8
        --long-request-sp-threshold "$LONG_REQUEST_SP_THRESHOLD"
        --long-request-sp-size "$LONG_REQUEST_SP_SIZE"
        --diagnostic-log-interval "$DIAGNOSTIC_LOG_INTERVAL"
        --slow-add-threshold-ms "$SLOW_ADD_THRESHOLD_MS"
        --run-label "${RUN_LABEL}_${NANODEPLOY_HIER_WORKER_TRANSPORT}"
        "$rate"
    )

    if (( DRY_RUN != 0 )); then
        log "DRY_RUN [$RUN_INDEX/$TOTAL_RUNS] dataset=$dataset_key rate=$rate num_requests=$num_requests"
        print_command "$stage_log_dir" "${cmd[@]}" | tee -a "$PROGRESS_LOG"
        started_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        record_summary "$dataset_key" "$rate" "$num_requests" "dry_run" "0" "0" "$started_at" "$started_at" "$stage_log_dir"
        return 0
    fi

    started_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    for ((attempt = 1; attempt <= MAX_RETRIES; attempt++)); do
        log "START [$RUN_INDEX/$TOTAL_RUNS] dataset=$dataset_key rate=$rate num_requests=$num_requests attempt=$attempt/$MAX_RETRIES"
        print_command "$stage_log_dir" "${cmd[@]}" >> "$console_log"

        set +e
        env BASE_LOG_DIR="$stage_log_dir" "${cmd[@]}" 2>&1 | tee -a "$console_log"
        rc="${PIPESTATUS[0]}"
        set -e

        if (( rc == 0 )); then
            set +e
            validation_output="$(python "$VALIDATE_RUN_PY" "$stage_log_dir" "$num_requests" "$NANODEPLOY_HIER_WORKER_TRANSPORT" 2>&1)"
            rc=$?
            set -e
            if (( rc == 0 )); then
                printf "completed_at_utc=%s\n%s\n" \
                    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$validation_output" > "$success_marker"
                break
            fi
            log "WARN dataset=$dataset_key rate=$rate result_validation_failed: $validation_output"
        fi

        log "WARN dataset=$dataset_key rate=$rate attempt=$attempt failed exit_code=$rc"
        if (( attempt < MAX_RETRIES )); then
            sleep "$RETRY_SLEEP_SECONDS"
        fi
    done

    finished_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    if (( rc == 0 )); then
        log "DONE [$RUN_INDEX/$TOTAL_RUNS] dataset=$dataset_key rate=$rate status=success $validation_output"
        record_summary "$dataset_key" "$rate" "$num_requests" "success" "0" "$attempt" "$started_at" "$finished_at" "$stage_log_dir"
        if (( RUN_INDEX < TOTAL_RUNS )); then
            sleep "$SLEEP_BETWEEN_RUNS"
        fi
        return 0
    fi

    log "FAILED [$RUN_INDEX/$TOTAL_RUNS] dataset=$dataset_key rate=$rate attempts=$MAX_RETRIES exit_code=$rc"
    record_summary "$dataset_key" "$rate" "$num_requests" "failed" "$rc" "$MAX_RETRIES" "$started_at" "$finished_at" "$stage_log_dir"
    return 1
}

run_dataset() {
    local dataset_key="$1"
    local dataset_path
    local rates=()
    local rate

    case "$dataset_key" in
        issue001)
            dataset_path="$ISSUE001_DATASET"
            rates=("${ISSUE001_RATE_LIST[@]}")
            ;;
        issue005)
            dataset_path="$ISSUE005_DATASET"
            rates=("${ISSUE005_RATE_LIST[@]}")
            ;;
        *)
            die "Unknown dataset key '$dataset_key'. Use issue001 and/or issue005."
            ;;
    esac

    for rate in "${rates[@]}"; do
        if ! run_one "$dataset_key" "$dataset_path" "$rate"; then
            OVERALL_RC=1
            if (( CONTINUE_ON_ERROR == 0 )); then
                return 1
            fi
            if (( RUN_INDEX < TOTAL_RUNS )); then
                sleep "$SLEEP_BETWEEN_RUNS"
            fi
        fi
    done
}

for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        usage
        exit 0
    fi
    [[ "$arg" != --* ]] || die "Unknown option '$arg'."
done

if (( $# == 0 )); then
    DATASET_KEYS=(issue001 issue005)
else
    DATASET_KEYS=("$@")
fi

validate_positive_integer DURATION_SECONDS "$DURATION_SECONDS"
validate_positive_integer BATCH_SIZE "$BATCH_SIZE"
validate_positive_integer SEGMENT_SIZE "$SEGMENT_SIZE"
validate_positive_integer MAX_MODEL_LEN "$MAX_MODEL_LEN"
validate_positive_integer MAX_INPUT_LEN "$MAX_INPUT_LEN"
validate_positive_integer MAX_REQUEST_TOKENS "$MAX_REQUEST_TOKENS"
validate_positive_integer LOOP_COUNT "$LOOP_COUNT"
validate_positive_integer LONG_REQUEST_SP_THRESHOLD "$LONG_REQUEST_SP_THRESHOLD"
validate_positive_integer LONG_REQUEST_SP_SIZE "$LONG_REQUEST_SP_SIZE"
validate_positive_integer MAX_RETRIES "$MAX_RETRIES"
validate_nonnegative_number GPU_MEMORY_LIMIT_GB "$GPU_MEMORY_LIMIT_GB"
validate_nonnegative_number GPU_MEMORY_UTILIZATION "$GPU_MEMORY_UTILIZATION"
validate_nonnegative_number DIAGNOSTIC_LOG_INTERVAL "$DIAGNOSTIC_LOG_INTERVAL"
validate_nonnegative_number SLOW_ADD_THRESHOLD_MS "$SLOW_ADD_THRESHOLD_MS"
validate_nonnegative_number RETRY_SLEEP_SECONDS "$RETRY_SLEEP_SECONDS"
validate_nonnegative_number SLEEP_BETWEEN_RUNS "$SLEEP_BETWEEN_RUNS"
validate_toggle DRY_RUN "$DRY_RUN"
validate_toggle RESUME "$RESUME"
validate_toggle FORCE_RERUN "$FORCE_RERUN"
validate_toggle CONTINUE_ON_ERROR "$CONTINUE_ON_ERROR"
validate_rates issue001 "${ISSUE001_RATE_LIST[@]}"
validate_rates issue005 "${ISSUE005_RATE_LIST[@]}"

(( EP_SIZE == 32 )) || die "Topology must produce EP=32; got EP=$EP_SIZE."
(( LONG_REQUEST_SP_SIZE <= SP_SIZE )) || die "LONG_REQUEST_SP_SIZE must be <= SP_SIZE ($SP_SIZE)."
[[ "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN_TAG contains unsafe path characters: '$RUN_TAG'."
[[ "$RUN_LABEL" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN_LABEL contains unsafe path characters: '$RUN_LABEL'."
[[ "$RAY_ADDR" =~ ^[^[:space:]:]+:[0-9]+$ ]] || die "RAY_ADDR must be host:port; got '$RAY_ADDR'."
[[ "$MASTER_ADDR" =~ ^[^[:space:]:]+:[0-9]+$ ]] || die "MASTER_ADDR must be host:port; got '$MASTER_ADDR'."

case "$ROUTING_STRATEGY" in
    LeastBatch|LeastCache|RoundRobin|VLLMLoadBalance) ;;
    *) die "Unsupported ROUTING_STRATEGY '$ROUTING_STRATEGY'." ;;
esac
case "$ROUTER_POLICY" in
    round_robin|least_batch|least_batch_v2|least_cache) ;;
    *) die "Unsupported ROUTER_POLICY '$ROUTER_POLICY'." ;;
esac
case "$SP_MASTER_SELECTOR" in
    RoundRobin|LeastBatch|LeastCache) ;;
    *) die "Unsupported SP_MASTER_SELECTOR '$SP_MASTER_SELECTOR'." ;;
esac
case "$SP_BACKEND" in
    legacy_ll|hao_basic|nccl) ;;
    *) die "Unsupported SP_BACKEND '$SP_BACKEND'." ;;
esac
case "$CUDA_GRAPH_MODE" in
    full|piecewise) ;;
    *) die "Unsupported CUDA_GRAPH_MODE '$CUDA_GRAPH_MODE'." ;;
esac

export NANODEPLOY_HIER_WORKER_TRANSPORT="${NANODEPLOY_HIER_WORKER_TRANSPORT:-zmq}"
case "$NANODEPLOY_HIER_WORKER_TRANSPORT" in
    ray|zmq) ;;
    *) die "NANODEPLOY_HIER_WORKER_TRANSPORT must be ray or zmq." ;;
esac

# Ray traffic must bypass HTTP proxies. QP=4 is explicitly propagated by the
# July RayExecutor to all ModelRunner actors.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export SLIME_QP_NUM=4
export DEEPEP_SMS="${DEEPEP_SMS:-16}"
export DEEPEP_MAX_TOKENS_PER_RANK="${DEEPEP_MAX_TOKENS_PER_RANK:-$BATCH_SIZE}"
export DEEPEP_ENABLE_MNNVL="${DEEPEP_ENABLE_MNNVL:-0}"
export DEEPEP_MODE=auto
export NANODEPLOY_HIER_RESULT_FASTPATH=0
export NANODEPLOY_LOG_DECODE_STEP_DETAIL=0
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export SLIME_VISIBLE_DEVICES="${SLIME_VISIBLE_DEVICES:-mlx5_bond_0}"
export SLIME_GID_INDEX="${SLIME_GID_INDEX:-3}"

verify_source_invariants

if (( DRY_RUN == 0 )); then
    [[ -d "$MODEL_PATH" ]] || die "Model directory not found: $MODEL_PATH"
    [[ -f "$ISSUE001_DATASET" ]] || die "Issue-1% dataset not found: $ISSUE001_DATASET"
    [[ -f "$ISSUE005_DATASET" ]] || die "Issue-5% dataset not found: $ISSUE005_DATASET"
fi

TOTAL_RUNS=0
for dataset_key in "${DATASET_KEYS[@]}"; do
    case "$dataset_key" in
        issue001) TOTAL_RUNS=$((TOTAL_RUNS + ${#ISSUE001_RATE_LIST[@]})) ;;
        issue005) TOTAL_RUNS=$((TOTAL_RUNS + ${#ISSUE005_RATE_LIST[@]})) ;;
        *) die "Unknown dataset key '$dataset_key'. Use issue001 and/or issue005." ;;
    esac
done

mkdir -p "$RUN_DIR"
PROGRESS_LOG="$RUN_DIR/matrix.progress"
SUMMARY_TSV="$RUN_DIR/matrix_summary.tsv"
EFFECTIVE_CONFIG="$RUN_DIR/effective_config.tsv"

if [[ ! -f "$SUMMARY_TSV" ]]; then
    printf "dataset\trate\tduration_seconds\tnum_requests\tstatus\texit_code\tattempts\tstarted_at_utc\tfinished_at_utc\tlog_dir\n" > "$SUMMARY_TSV"
fi

CONFIG_TMP="$(mktemp /tmp/nanodeploy-decent-e2e-config.XXXXXX)"
trap 'rm -f "$CONFIG_TMP"' EXIT
write_effective_config > "$CONFIG_TMP"
if [[ -f "$EFFECTIVE_CONFIG" ]]; then
    if ! cmp -s "$CONFIG_TMP" "$EFFECTIVE_CONFIG"; then
        diff -u "$EFFECTIVE_CONFIG" "$CONFIG_TMP" || true
        die "RUN_TAG '$RUN_TAG' already exists with a different effective configuration. Choose a new RUN_TAG or restore the original values."
    fi
else
    cp "$CONFIG_TMP" "$EFFECTIVE_CONFIG"
fi

log "RUN_TAG=$RUN_TAG"
log "RUN_DIR=$RUN_DIR"
log "RAY_ADDR=$RAY_ADDR MASTER_ADDR=$MASTER_ADDR"
log "TOPOLOGY=DP${DP_SIZE}xSP${SP_SIZE}xTP${TP_SIZE}=EP${EP_SIZE} (4 nodes x 8 GPUs)"
log "WORKLOAD=duration:${DURATION_SECONDS}s batch:${BATCH_SIZE} max_model:${MAX_MODEL_LEN} max_input:${MAX_INPUT_LEN} max_request:${MAX_REQUEST_TOKENS}"
log "ENGINE=max_batched_tokens:${MAX_NUM_BATCHED_TOKENS} kv_block:${KV_CACHE_BLOCK_SIZE} max_recv:${MAX_NUM_RECV_SEQS} loop:${LOOP_COUNT}"
log "POLICY=hierarchical/${ROUTER_POLICY} routing:${ROUTING_STRATEGY} sp_master:${SP_MASTER_SELECTOR} dynamic:long_short_sp8 threshold:${LONG_REQUEST_SP_THRESHOLD} long_sp:${LONG_REQUEST_SP_SIZE}"
log "BACKEND=${SP_BACKEND}/${CUDA_GRAPH_MODE} worker_transport:${NANODEPLOY_HIER_WORKER_TRANSPORT} SLIME_QP_NUM:${SLIME_QP_NUM} DEEPEP_SMS:${DEEPEP_SMS} DEEPEP_MAX_TOKENS_PER_RANK:${DEEPEP_MAX_TOKENS_PER_RANK} DEEPEP_ENABLE_MNNVL:${DEEPEP_ENABLE_MNNVL} DEEPEP_MODE:${DEEPEP_MODE}"
log "DATASETS=${DATASET_KEYS[*]} ISSUE001_RATES=${ISSUE001_RATES} ISSUE005_RATES=${ISSUE005_RATES} TOTAL_RUNS=$TOTAL_RUNS"
log "DRY_RUN=$DRY_RUN RESUME=$RESUME FORCE_RERUN=$FORCE_RERUN CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR MAX_RETRIES=$MAX_RETRIES"

RUN_INDEX=0
OVERALL_RC=0
for dataset_key in "${DATASET_KEYS[@]}"; do
    if ! run_dataset "$dataset_key"; then
        OVERALL_RC=1
        break
    fi
done

if (( OVERALL_RC == 0 )); then
    log "ALL_DONE status=success"
else
    log "ALL_DONE status=failed"
fi
exit "$OVERALL_RC"
