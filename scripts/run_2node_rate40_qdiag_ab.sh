#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
START_BENCH_SH="$ROOT_DIR/scripts/sp_ablation/start_bench.sh"
ANALYZER="$ROOT_DIR/utils_analysis/analyze_2node_qdiag_ab.py"

RAY_ADDR="${RAY_ADDR:-10.102.243.60:6380}"
MASTER_ADDR="${MASTER_ADDR:-10.102.243.60:29500}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--deepseek-ai--DeepSeek-V3/snapshots/e815299b0bcbac849fa540c768ef21845365c9eb}"
DATASET_PATH="${DATASET_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv}"

RATE="${RATE:-40}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
if (( SMOKE != 0 )); then
    DURATION_SECONDS="${DURATION_SECONDS:-30}"
    DEFAULT_RUN_ORDER="central hierarchical"
else
    DURATION_SECONDS="${DURATION_SECONDS:-300}"
    DEFAULT_RUN_ORDER="hierarchical central central hierarchical hierarchical central"
fi
NUM_REQUESTS="${NUM_REQUESTS:-$(awk -v rate="$RATE" -v seconds="$DURATION_SECONDS" 'BEGIN {printf "%d", rate * seconds}')}"
RUN_ORDER="${RUN_ORDER:-$DEFAULT_RUN_ORDER}"
read -r -a STAGES <<< "$RUN_ORDER"

SEGMENT_SIZE="${SEGMENT_SIZE:-65536}"
BATCH_SIZE="${BATCH_SIZE:-192}"
GPU_MEMORY_LIMIT_GB="${GPU_MEMORY_LIMIT_GB:-141}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
MAX_REQUEST_TOKENS="${MAX_REQUEST_TOKENS:-910000}"
LOOP_COUNT="${LOOP_COUNT:-16}"
ROUTER_POLICY="${ROUTER_POLICY:-least_batch}"
# Keep periodic scheduler snapshots off for the paired run: only the common
# quantum stream is enabled, avoiding asymmetric hierarchical polling work.
DIAGNOSTIC_LOG_INTERVAL="${DIAGNOSTIC_LOG_INTERVAL:-0}"
RUN_TAG="${RUN_TAG:-two_node_r40_qdiag_ab_$(date -u +%Y%m%d_%H%M%S)}"
CHAIN_LOG_DIR="${CHAIN_LOG_DIR:-$ROOT_DIR/bench_logs/$RUN_TAG}"
PROGRESS_LOG="$CHAIN_LOG_DIR/ab.progress"
MANIFEST_TSV="$CHAIN_LOG_DIR/run_manifest.tsv"

export SLIME_VISIBLE_DEVICES="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"
export SLIME_GID_INDEX=3
export SLIME_QP_NUM=4
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

log() {
    local now
    now="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "[$now] $*" | tee -a "$PROGRESS_LOG"
}

print_command() {
    local stage_log_dir="$1"
    shift
    printf "env BASE_LOG_DIR=%q " "$stage_log_dir"
    printf "%q " "$@"
    printf "\n"
}

stage_config() {
    case "$1" in
        central)
            STAGE_ARCH="legacy_global"
            ;;
        hierarchical)
            STAGE_ARCH="hierarchical"
            ;;
        *)
            echo "Error: RUN_ORDER contains unknown stage '$1'." >&2
            return 2
            ;;
    esac
}

for value_name in SMOKE DRY_RUN CONTINUE_ON_ERROR; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[01]$ ]]; then
        echo "Error: $value_name must be 0 or 1." >&2
        exit 2
    fi
done
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
if (( ${#STAGES[@]} == 0 )); then
    echo "Error: RUN_ORDER must contain at least one stage." >&2
    exit 2
fi
for stage in "${STAGES[@]}"; do
    stage_config "$stage"
done
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

mkdir -p "$CHAIN_LOG_DIR"
printf "run_index\tstage\tscheduler_arch\tstatus\texit_code\tstarted_at_utc\tlog_dir\tsummary_json\n" > "$MANIFEST_TSV"

log "RUN_TAG=$RUN_TAG"
log "RAY_ADDR=$RAY_ADDR MASTER_ADDR=$MASTER_ADDR"
log "MODEL_PATH=$MODEL_PATH"
log "DATASET_PATH=$DATASET_PATH"
log "RUN_ORDER=$RUN_ORDER"
log "RATE=$RATE DURATION_SECONDS=$DURATION_SECONDS NUM_REQUESTS=$NUM_REQUESTS"
log "ROUTER_POLICY=$ROUTER_POLICY"
log "SLIME_VISIBLE_DEVICES=$SLIME_VISIBLE_DEVICES SLIME_GID_INDEX=$SLIME_GID_INDEX SLIME_QP_NUM=$SLIME_QP_NUM"
log "DRY_RUN=$DRY_RUN SMOKE=$SMOKE CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR"

overall_rc=0
run_index=0
for stage in "${STAGES[@]}"; do
    run_index=$((run_index + 1))
    stage_config "$stage"
    run_id="$(printf '%02d_%s' "$run_index" "$stage")"
    stage_log_dir="$CHAIN_LOG_DIR/$run_id"
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
        --dp-size 2
        --sp-size 8
        --tp-size 1
        --batch-size "$BATCH_SIZE"
        --num-requests "$NUM_REQUESTS"
        --gpu-mem "$GPU_MEMORY_LIMIT_GB"
        --gpu-util "$GPU_MEMORY_UTILIZATION"
        --max-model-len "$MAX_MODEL_LEN"
        --max-request-tokens "$MAX_REQUEST_TOKENS"
        --routing-strategy LeastBatch
        --scheduler-arch "$STAGE_ARCH"
        --router-policy "$ROUTER_POLICY"
        --sp-master-selector LeastBatch
        --loop-count "$LOOP_COUNT"
        --fixed-sp-size 0
        --sp-backend hao_basic
        --cuda-graph-mode full
        --diagnostic-log-interval "$DIAGNOSTIC_LOG_INTERVAL"
        --slow-add-threshold-ms 20
        --quantum-diagnostics
        --run-label "qdiag_${run_id}"
        "$RATE"
    )

    log "START run=$run_id scheduler_arch=$STAGE_ARCH"
    if (( DRY_RUN != 0 )); then
        print_command "$stage_log_dir" "${cmd[@]}" | tee -a "$PROGRESS_LOG"
        printf "%s\t%s\t%s\tdry_run\t0\t%s\t%s\t\n" \
            "$run_id" "$stage" "$STAGE_ARCH" "$started_at" \
            "$stage_log_dir" >> "$MANIFEST_TSV"
        continue
    fi

    set +e
    env BASE_LOG_DIR="$stage_log_dir" "${cmd[@]}" 2>&1 | tee -a "$console_log"
    rc="${PIPESTATUS[0]}"
    set -e
    summary_json=""
    if (( rc == 0 )); then
        summary_json="$(find "$stage_log_dir" -type f -name '*.summary.json' -print -quit)"
        if [[ -z "$summary_json" ]]; then
            log "VALIDATION_FAILED run=$run_id reason=missing_summary"
            rc=3
        fi
    fi
    if (( rc == 0 )); then
        status="success"
        log "DONE run=$run_id status=success summary=$summary_json"
    else
        status="failed"
        overall_rc=1
        log "DONE run=$run_id status=failed exit_code=$rc"
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$run_id" "$stage" "$STAGE_ARCH" "$status" "$rc" \
        "$started_at" "$stage_log_dir" "$summary_json" \
        >> "$MANIFEST_TSV"
    if (( rc != 0 && CONTINUE_ON_ERROR == 0 )); then
        break
    fi
done

if (( DRY_RUN != 0 )); then
    log "DRY_RUN_DONE manifest=$MANIFEST_TSV"
elif (( overall_rc == 0 )); then
    log "RUNS_DONE manifest=$MANIFEST_TSV"
    if [[ -f "$ANALYZER" ]]; then
        python3 "$ANALYZER" --manifest "$MANIFEST_TSV"
    fi
fi

exit "$overall_rc"
