#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RATE_SWEEP_SCRIPT="$SCRIPT_DIR/run_kimi_issue001_4node_dp4sp8_longshort_10min_rates.sh"

DPSK_MODEL_PATH="${DPSK_MODEL_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3}"
KIMI_MODEL_PATH="${KIMI_MODEL_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/Kimi-K2-Instruct-0905}"

DATASET_ROOT="${DATASET_ROOT:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset}"
GEMINI_ISSUES_DATASET="${GEMINI_ISSUES_DATASET:-$DATASET_ROOT/madha/Gemini_Issues_Stats_rename-shuffle.csv}"
KIMI_ISSUE001_DATASET="${KIMI_ISSUE001_DATASET:-$DATASET_ROOT/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv}"
KIMI_ISSUE005_DATASET="${KIMI_ISSUE005_DATASET:-$DATASET_ROOT/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.05_n60000_60k.csv}"

# Keep these as five stages so the requested cross-dataset ordering is exact.
DPSK_GEMINI_RATES="${DPSK_GEMINI_RATES:-0.25 0.5 1 1.5 1.75 2 2.5}"
KIMI_ISSUE001_FIRST_RATES="${KIMI_ISSUE001_FIRST_RATES:-10 20 30 40 50}"
KIMI_ISSUE005_FIRST_RATES="${KIMI_ISSUE005_FIRST_RATES:-2.5 10 20 25 30}"
KIMI_ISSUE001_SECOND_RATES="${KIMI_ISSUE001_SECOND_RATES:-60 70 80 90 100 110}"
KIMI_ISSUE005_SECOND_RATES="${KIMI_ISSUE005_SECOND_RATES:-35 40 45 50}"

SEND_DURATION_SEC="${SEND_DURATION_SEC:-600}"
MAX_REQUEST_TOKENS="${MAX_REQUEST_TOKENS:-0}"
DRY_RUN="${DRY_RUN:-0}"

RUN_TAG="${RUN_TAG:-dpsk_gemini_kimi_issue_rates_$(date -u +%Y%m%d_%H%M%S)}"
E2E_LOG_ROOT="${E2E_LOG_ROOT:-$REPO_ROOT/bench_logs/e2e}"
OUTPUT_DIR="${OUTPUT_DIR:-$E2E_LOG_ROOT/$RUN_TAG}"
PROGRESS_LOG="$OUTPUT_DIR/ordered_run.progress"
SUMMARY_FILE="$OUTPUT_DIR/ordered_run_summary.tsv"
CONFIG_FILE="$OUTPUT_DIR/ordered_run_config.txt"

log() {
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*" \
        | tee -a "$PROGRESS_LOG"
}

require_dir() {
    local label="$1"
    local path="$2"
    if [[ ! -d "$path" ]]; then
        printf 'Missing %s directory: %s\n' "$label" "$path" >&2
        exit 1
    fi
}

require_file() {
    local label="$1"
    local path="$2"
    if [[ ! -f "$path" ]]; then
        printf 'Missing %s file: %s\n' "$label" "$path" >&2
        exit 1
    fi
}

run_stage() {
    local stage_key="$1"
    local model_key="$2"
    local dataset_key="$3"
    local model_path="$4"
    local dataset_path="$5"
    local rates="$6"
    local stage_output_dir="$OUTPUT_DIR/$stage_key"
    local stage_run_tag="${RUN_TAG}_${stage_key}"
    local status="ok"
    local rc=0

    log "START stage=$stage_key model=$model_key dataset=$dataset_key rates=$rates"

    set +e
    env \
        MODEL_PATH="$model_path" \
        DATASET_PATH="$dataset_path" \
        REQUEST_RATES="$rates" \
        SEND_DURATION_SEC="$SEND_DURATION_SEC" \
        MAX_REQUEST_TOKENS="$MAX_REQUEST_TOKENS" \
        RUN_TAG="$stage_run_tag" \
        OUTPUT_DIR="$stage_output_dir" \
        DRY_RUN="$DRY_RUN" \
        SLIME_QP_NUM=4 \
        bash "$RATE_SWEEP_SCRIPT"
    rc=$?
    set -e

    if [[ "$rc" -ne 0 ]]; then
        status="failed"
    elif [[ "$DRY_RUN" == "1" ]]; then
        status="planned"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$stage_key" "$model_key" "$dataset_key" "$rates" "$status" "$stage_output_dir" \
        >> "$SUMMARY_FILE"

    if [[ "$rc" -ne 0 ]]; then
        log "FAILED stage=$stage_key status=$rc; stopping ordered run"
        exit "$rc"
    fi

    log "DONE stage=$stage_key status=$status"
}

require_file "rate sweep script" "$RATE_SWEEP_SCRIPT"
require_dir "DPSK model" "$DPSK_MODEL_PATH"
require_dir "KIMI model" "$KIMI_MODEL_PATH"
require_file "Gemini Issues dataset" "$GEMINI_ISSUES_DATASET"
require_file "KIMI Issue1% dataset" "$KIMI_ISSUE001_DATASET"
require_file "KIMI Issue5% dataset" "$KIMI_ISSUE005_DATASET"

if [[ -e "$OUTPUT_DIR" ]]; then
    printf 'Refusing to overwrite existing output directory: %s\n' "$OUTPUT_DIR" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

printf 'stage\tmodel\tdataset\trates\tstatus\toutput_dir\n' > "$SUMMARY_FILE"
{
    printf 'RUN_TAG=%q\n' "$RUN_TAG"
    printf 'DPSK_MODEL_PATH=%q\n' "$DPSK_MODEL_PATH"
    printf 'KIMI_MODEL_PATH=%q\n' "$KIMI_MODEL_PATH"
    printf 'GEMINI_ISSUES_DATASET=%q\n' "$GEMINI_ISSUES_DATASET"
    printf 'KIMI_ISSUE001_DATASET=%q\n' "$KIMI_ISSUE001_DATASET"
    printf 'KIMI_ISSUE005_DATASET=%q\n' "$KIMI_ISSUE005_DATASET"
    printf 'SEND_DURATION_SEC=%q\n' "$SEND_DURATION_SEC"
    printf 'MAX_REQUEST_TOKENS=%q\n' "$MAX_REQUEST_TOKENS"
    printf 'SLIME_QP_NUM=4\n'
    printf 'OUTPUT_DIR=%q\n' "$OUTPUT_DIR"
} > "$CONFIG_FILE"

log "START ordered run tag=$RUN_TAG duration=${SEND_DURATION_SEC}s max_request_tokens=$MAX_REQUEST_TOKENS"
log "ORDER=dpsk_gemini -> kimi_issue001_first -> kimi_issue005_first -> kimi_issue001_second -> kimi_issue005_second"

run_stage \
    "01_dpsk_gemini_issues" "dpsk" "gemini_issues" \
    "$DPSK_MODEL_PATH" "$GEMINI_ISSUES_DATASET" "$DPSK_GEMINI_RATES"
run_stage \
    "02_kimi_issue001_first" "kimi" "issue001" \
    "$KIMI_MODEL_PATH" "$KIMI_ISSUE001_DATASET" "$KIMI_ISSUE001_FIRST_RATES"
run_stage \
    "03_kimi_issue005_first" "kimi" "issue005" \
    "$KIMI_MODEL_PATH" "$KIMI_ISSUE005_DATASET" "$KIMI_ISSUE005_FIRST_RATES"
run_stage \
    "04_kimi_issue001_second" "kimi" "issue001" \
    "$KIMI_MODEL_PATH" "$KIMI_ISSUE001_DATASET" "$KIMI_ISSUE001_SECOND_RATES"
run_stage \
    "05_kimi_issue005_second" "kimi" "issue005" \
    "$KIMI_MODEL_PATH" "$KIMI_ISSUE005_DATASET" "$KIMI_ISSUE005_SECOND_RATES"

if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN complete; all 27 rates were planned and no benchmark was launched"
else
    log "COMPLETE all 27 rates finished in the requested order"
fi
