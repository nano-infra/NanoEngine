#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
START_BENCH_SH="$ROOT_DIR/scripts/sp_ablation/start_bench.sh"

RAY_ADDR="${RAY_ADDR:-10.102.206.14:8776}"
MASTER_ADDR="${MASTER_ADDR:-10.102.206.14:29500}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme1n1/ml_research/chenjiefei/models/deepseek-v3}"
DATASET_PATH="${DATASET_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv}"

RATE="${RATE:-30}"
DURATION_SECONDS="${DURATION_SECONDS:-360}"
NUM_REQUESTS="${NUM_REQUESTS:-$(awk -v rate="$RATE" -v seconds="$DURATION_SECONDS" 'BEGIN {printf "%d", rate * seconds}')}"

SEGMENT_SIZE="${SEGMENT_SIZE:-65536}"
BATCH_SIZE="${BATCH_SIZE:-192}"
GPU_MEMORY_LIMIT_GB="${GPU_MEMORY_LIMIT_GB:-141}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
MAX_REQUEST_TOKENS="${MAX_REQUEST_TOKENS:-910000}"
LOOP_COUNT="${LOOP_COUNT:-16}"
SP_BACKEND="hao_basic"
CUDA_GRAPH_MODE="${CUDA_GRAPH_MODE:-full}"
ROUTING_STRATEGY="${ROUTING_STRATEGY:-LeastBatch}"
ROUTER_POLICY="${ROUTER_POLICY:-least_batch}"
SP_MASTER_SELECTOR="${SP_MASTER_SELECTOR:-LeastBatch}"
DIAGNOSTIC_LOG_INTERVAL="${DIAGNOSTIC_LOG_INTERVAL:-1}"
SLOW_ADD_THRESHOLD_MS="${SLOW_ADD_THRESHOLD_MS:-20}"
HIERARCHICAL_QUANTUM_DIAGNOSTICS="${HIERARCHICAL_QUANTUM_DIAGNOSTICS:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
RUN_LABEL="${RUN_LABEL:-rate30_6min}"
RUN_TAG="${RUN_TAG:-2node_rate30_6min_matrix_$(date -u +%Y%m%d_%H%M%S)}"
CHAIN_LOG_DIR="${CHAIN_LOG_DIR:-$ROOT_DIR/bench_logs/$RUN_TAG}"
PROGRESS_LOG="$CHAIN_LOG_DIR/matrix.progress"
SUMMARY_TSV="$CHAIN_LOG_DIR/matrix_summary.tsv"

DEFAULT_STAGES=(
    centralized_dp16
    centralized_dp2sp8
    decentralized_dp16
    decentralized_dp2sp8
)

if (( $# > 0 )); then
    STAGES=("$@")
else
    STAGES=("${DEFAULT_STAGES[@]}")
fi

log() {
    local now
    now="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "[$now] $*" | tee -a "$PROGRESS_LOG"
}

usage() {
    echo "Usage: $0 [stage ...]"
    echo
    echo "Stages:"
    echo "  centralized_dp16"
    echo "  centralized_dp2sp8"
    echo "  decentralized_dp16"
    echo "  decentralized_dp2sp8"
    echo
    echo "No stage arguments runs all four in the order above."
    echo "Set DRY_RUN=1 to print commands without launching the benchmark."
}

stage_config() {
    local stage="$1"
    case "$stage" in
        centralized_dp16)
            STAGE_SCHEDULER_ARCH="legacy_global"
            STAGE_DP=16
            STAGE_SP=1
            ;;
        centralized_dp2sp8)
            STAGE_SCHEDULER_ARCH="legacy_global"
            STAGE_DP=2
            STAGE_SP=8
            ;;
        decentralized_dp16)
            STAGE_SCHEDULER_ARCH="hierarchical"
            STAGE_DP=16
            STAGE_SP=1
            ;;
        decentralized_dp2sp8)
            STAGE_SCHEDULER_ARCH="hierarchical"
            STAGE_DP=2
            STAGE_SP=8
            ;;
        *)
            echo "Error: unknown stage '$stage'." >&2
            usage >&2
            return 1
            ;;
    esac
}

record_summary() {
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$1" "$2" "$3" "$4" "$RATE" "$DURATION_SECONDS" "$NUM_REQUESTS" \
        "$5" "$6" "$7" "$8" >> "$SUMMARY_TSV"
}

print_command() {
    local stage_log_dir="$1"
    shift
    printf "env BASE_LOG_DIR=%q " "$stage_log_dir"
    printf "%q " "$@"
    printf "\n"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if ! [[ "$RATE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error: RATE must be a positive number." >&2
    exit 2
fi
if ! [[ "$DURATION_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: DURATION_SECONDS must be a positive integer." >&2
    exit 2
fi
if ! [[ "$NUM_REQUESTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_REQUESTS must be a positive integer." >&2
    exit 2
fi
if ! [[ "$MAX_REQUEST_TOKENS" =~ ^[0-9]+$ ]]; then
    echo "Error: MAX_REQUEST_TOKENS must be a non-negative integer." >&2
    exit 2
fi
if ! [[ "$DRY_RUN" =~ ^[01]$ && "$CONTINUE_ON_ERROR" =~ ^[01]$ && "$HIERARCHICAL_QUANTUM_DIAGNOSTICS" =~ ^[01]$ ]]; then
    echo "Error: DRY_RUN, CONTINUE_ON_ERROR, and HIERARCHICAL_QUANTUM_DIAGNOSTICS must be 0 or 1." >&2
    exit 2
fi
if [[ ! -f "$START_BENCH_SH" ]]; then
    echo "Error: start_bench.sh not found: $START_BENCH_SH" >&2
    exit 2
fi
if (( DRY_RUN == 0 )); then
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo "Error: model directory not found: $MODEL_PATH" >&2
        exit 2
    fi
    if [[ ! -f "$DATASET_PATH" ]]; then
        echo "Error: dataset not found: $DATASET_PATH" >&2
        exit 2
    fi
fi

for stage in "${STAGES[@]}"; do
    stage_config "$stage" || exit 2
done

mkdir -p "$CHAIN_LOG_DIR"
if [[ ! -f "$SUMMARY_TSV" ]]; then
    printf "stage\tscheduler_arch\tdp\tsp\trate\tduration_seconds\tnum_requests\tstatus\texit_code\tstarted_at_utc\tlog_dir\n" \
        > "$SUMMARY_TSV"
fi

log "RUN_TAG=$RUN_TAG"
log "RAY_ADDR=$RAY_ADDR MASTER_ADDR=$MASTER_ADDR"
log "MODEL_PATH=$MODEL_PATH"
log "DATASET_PATH=$DATASET_PATH"
log "RATE=$RATE DURATION_SECONDS=$DURATION_SECONDS NUM_REQUESTS=$NUM_REQUESTS"
log "MAX_REQUEST_TOKENS=$MAX_REQUEST_TOKENS"
log "HIERARCHICAL_QUANTUM_DIAGNOSTICS=$HIERARCHICAL_QUANTUM_DIAGNOSTICS"
log "ROUTING_STRATEGY=$ROUTING_STRATEGY ROUTER_POLICY=$ROUTER_POLICY SP_MASTER_SELECTOR=$SP_MASTER_SELECTOR"
log "STAGES=${STAGES[*]}"
log "DRY_RUN=$DRY_RUN CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR"

overall_rc=0

for stage in "${STAGES[@]}"; do
    stage_config "$stage"
    stage_log_dir="$CHAIN_LOG_DIR/$stage"
    console_log="$stage_log_dir/console.log"
    started_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    mkdir -p "$stage_log_dir"

    cmd=(
        bash "$START_BENCH_SH"
        --master-addr "$MASTER_ADDR"
        --ray-addr "$RAY_ADDR"
        --dataset-path "$DATASET_PATH"
        --model-path "$MODEL_PATH"
        --segment-size "$SEGMENT_SIZE"
        --dp-size "$STAGE_DP"
        --sp-size "$STAGE_SP"
        --tp-size 1
        --batch-size "$BATCH_SIZE"
        --num-requests "$NUM_REQUESTS"
        --gpu-mem "$GPU_MEMORY_LIMIT_GB"
        --gpu-util "$GPU_MEMORY_UTILIZATION"
        --max-model-len "$MAX_MODEL_LEN"
        --max-request-tokens "$MAX_REQUEST_TOKENS"
        --routing-strategy "$ROUTING_STRATEGY"
        --scheduler-arch "$STAGE_SCHEDULER_ARCH"
        --router-policy "$ROUTER_POLICY"
        --sp-master-selector "$SP_MASTER_SELECTOR"
        --loop-count "$LOOP_COUNT"
        --fixed-sp-size 0
        --sp-backend "$SP_BACKEND"
        --cuda-graph-mode "$CUDA_GRAPH_MODE"
        --diagnostic-log-interval "$DIAGNOSTIC_LOG_INTERVAL"
        --slow-add-threshold-ms "$SLOW_ADD_THRESHOLD_MS"
        --run-label "${RUN_LABEL}_${stage}"
    )
    if (( ENFORCE_EAGER != 0 )); then
        cmd+=(--enforce-eager)
    fi
    if [[ "$STAGE_SCHEDULER_ARCH" == "hierarchical" ]] && (( HIERARCHICAL_QUANTUM_DIAGNOSTICS != 0 )); then
        cmd+=(--hierarchical-quantum-diagnostics)
    fi
    cmd+=("$RATE")

    log "START stage=$stage scheduler_arch=$STAGE_SCHEDULER_ARCH dp=$STAGE_DP sp=$STAGE_SP"
    if (( DRY_RUN != 0 )); then
        print_command "$stage_log_dir" "${cmd[@]}" | tee -a "$PROGRESS_LOG"
        record_summary \
            "$stage" "$STAGE_SCHEDULER_ARCH" "$STAGE_DP" "$STAGE_SP" \
            "dry_run" "0" "$started_at" "$stage_log_dir"
        continue
    fi

    set +e
    env BASE_LOG_DIR="$stage_log_dir" "${cmd[@]}" 2>&1 | tee -a "$console_log"
    rc="${PIPESTATUS[0]}"
    set -e

    if (( rc == 0 )); then
        status="success"
        log "DONE stage=$stage status=$status"
    else
        status="failed"
        overall_rc=1
        log "DONE stage=$stage status=$status exit_code=$rc"
    fi
    record_summary \
        "$stage" "$STAGE_SCHEDULER_ARCH" "$STAGE_DP" "$STAGE_SP" \
        "$status" "$rc" "$started_at" "$stage_log_dir"

    if (( rc != 0 && CONTINUE_ON_ERROR == 0 )); then
        log "STOP reason=stage_failed stage=$stage"
        break
    fi
done

if (( DRY_RUN != 0 )); then
    log "DRY_RUN_DONE"
elif (( overall_rc == 0 )); then
    log "ALL_DONE"
fi

exit "$overall_rc"
