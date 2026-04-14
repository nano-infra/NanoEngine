#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SWEEP_SCRIPT="$ROOT_DIR/scripts/issue003/run_issue003_deepseek_rate5_sweep.sh"

RAY_ADDR="${RAY_ADDR:-10.102.97.179:7799}"
MASTER_ADDR="${MASTER_ADDR:-10.102.97.179:29500}"

DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-/mnt/nvme1n1/ml_research/models/deepseek-v3}"
KIMI_MODEL="${KIMI_MODEL:-/mnt/nvme1n1/ml_research/models/models--moonshotai--Kimi-K2-Instruct-0905/snapshots/7152993552508c9f22042b3bb93b5e6acd06ce73}"

ISSUE003_DATASET="${ISSUE003_DATASET:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.03_n60000.csv}"
ISSUE001_DATASET="${ISSUE001_DATASET:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv}"
MIXED60K_DATASET="${MIXED60K_DATASET:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o/sharegpt4o-mixed-random-60k.csv}"
GEMINI_ISSUES_DATASET="${GEMINI_ISSUES_DATASET:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/madha/Gemini_Issues_Stats_rename-shuffle.csv}"

BATCH_SIZE="${BATCH_SIZE:-256}"
MIXED60K_BATCH_SIZE="${MIXED60K_BATCH_SIZE:-384}"
SEG="${SEG:-65536}"
SCHEDULER="${SCHEDULER:-centralized}"
GPU_MEM="${GPU_MEM:-141}"
GPU_UTIL="${GPU_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-1000000}"
LOOP_COUNT="${LOOP_COUNT:-16}"
FIXED_SP_SEGMENTS="${FIXED_SP_SEGMENTS:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
DP4_START_RATE="${DP4_START_RATE:-10}"
DP4_STEP_RATE="${DP4_STEP_RATE:-10}"
MIXED60K_START_RATE="${MIXED60K_START_RATE:-20}"
MIXED60K_STEP_RATE="${MIXED60K_STEP_RATE:-20}"
MIXED60K_RERUN_START_RATE="${MIXED60K_RERUN_START_RATE:-120}"
MIXED60K_RERUN_STEP_RATE="${MIXED60K_RERUN_STEP_RATE:-20}"
DP32_START_RATE="${DP32_START_RATE:-10}"
DP32_STEP_RATE="${DP32_STEP_RATE:-10}"
GEMINI_ISSUES_DP4_START_RATE="${GEMINI_ISSUES_DP4_START_RATE:-$DP4_START_RATE}"
GEMINI_ISSUES_DP4_STEP_RATE="${GEMINI_ISSUES_DP4_STEP_RATE:-2}"
GEMINI_ISSUES_DP32_START_RATE="${GEMINI_ISSUES_DP32_START_RATE:-$DP32_START_RATE}"
GEMINI_ISSUES_DP32_STEP_RATE="${GEMINI_ISSUES_DP32_STEP_RATE:-2}"
MAX_RATE="${MAX_RATE:-500}"
STOP_THRESHOLD_MS="${STOP_THRESHOLD_MS:-100}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-20}"
MAX_RETRIES="${MAX_RETRIES:-5}"
LONG_REQUEST_SP_THRESHOLD="${LONG_REQUEST_SP_THRESHOLD:-100000}"

RUN_TAG="${RUN_TAG:-issue001_issue003_mixed60k_longshort_dp4dp32_bs256_$(date -u +%Y%m%d_%H%M%S)}"
CHAIN_LOG_DIR="${CHAIN_LOG_DIR:-$ROOT_DIR/bench_logs/$RUN_TAG}"
CHAIN_PROGRESS="$CHAIN_LOG_DIR/chain.progress"
CHAIN_SUMMARY="$CHAIN_LOG_DIR/chain_summary.tsv"

mkdir -p "$CHAIN_LOG_DIR"

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "$CHAIN_PROGRESS"
}

ensure_headers() {
    if [[ ! -f "$CHAIN_SUMMARY" ]]; then
        printf "phase\tmodel_key\tdataset_key\tbase_log_dir\tstatus\n" > "$CHAIN_SUMMARY"
    fi
}

record_stage() {
    printf "%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$5" >> "$CHAIN_SUMMARY"
}

run_stage() {
    local phase="$1"
    local model_key="$2"
    local dataset_key="$3"
    local model_path="$4"
    local dataset_path="$5"
    local strategy="$6"
    local sweep_strategy="$7"
    local enable_dynamic_sp_size="$8"
    local start_rate="$9"
    local step_rate="${10}"
    local base_log_dir="${11}"
    local stage_batch_size="${12:-$BATCH_SIZE}"

    log "START phase=$phase model=$model_key dataset=$dataset_key strategy=$strategy sweep=$sweep_strategy dynamic_sp=$enable_dynamic_sp_size start_rate=$start_rate step_rate=$step_rate batch_size=$stage_batch_size"

    env \
        RAY_ADDR="$RAY_ADDR" \
        MASTER_ADDR="$MASTER_ADDR" \
        BATCH_SIZE="$stage_batch_size" \
        SEG="$SEG" \
        SCHEDULER="$SCHEDULER" \
        GPU_MEM="$GPU_MEM" \
        GPU_UTIL="$GPU_UTIL" \
        MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        MAX_INPUT_LEN="$MAX_INPUT_LEN" \
        LOOP_COUNT="$LOOP_COUNT" \
        FIXED_SP_SEGMENTS="$FIXED_SP_SEGMENTS" \
        ENFORCE_EAGER="$ENFORCE_EAGER" \
        START_RATE="$start_rate" \
        STEP_RATE="$step_rate" \
        MAX_RATE="$MAX_RATE" \
        STOP_THRESHOLD_MS="$STOP_THRESHOLD_MS" \
        SLEEP_BETWEEN_RUNS="$SLEEP_BETWEEN_RUNS" \
        MAX_RETRIES="$MAX_RETRIES" \
        MODEL_PATH="$model_path" \
        DATASET_PATH="$dataset_path" \
        BASE_LOG_DIR="$base_log_dir" \
        RUN_TAG="$(basename "$base_log_dir")" \
        ENABLE_DYNAMIC_SP_SIZE="$enable_dynamic_sp_size" \
        USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER=0 \
        DYNAMIC_SP_SIZE_STRATEGY="$strategy" \
        LONG_REQUEST_SP_THRESHOLD="$LONG_REQUEST_SP_THRESHOLD" \
        SWEEP_STRATEGY="$sweep_strategy" \
        bash "$SWEEP_SCRIPT"

    record_stage "$phase" "$model_key" "$dataset_key" "$base_log_dir" "done"
    log "DONE phase=$phase model=$model_key dataset=$dataset_key strategy=$strategy sweep=$sweep_strategy dynamic_sp=$enable_dynamic_sp_size start_rate=$start_rate step_rate=$step_rate batch_size=$stage_batch_size"
}

ensure_headers

log "CHAIN_TAG=$RUN_TAG"
log "RAY_ADDR=$RAY_ADDR MASTER_ADDR=$MASTER_ADDR"
log "BATCH_SIZE(default)=$BATCH_SIZE MIXED60K_BATCH_SIZE=$MIXED60K_BATCH_SIZE GPU_UTIL=$GPU_UTIL ENFORCE_EAGER=$ENFORCE_EAGER"
log "MIXED60K_RERUN_DP4_RATE=start:$MIXED60K_RERUN_START_RATE step:$MIXED60K_RERUN_STEP_RATE | ISSUE001_DP4_RATE=start:$DP4_START_RATE step:$DP4_STEP_RATE | ISSUE001_DP32_RATE=start:$DP32_START_RATE step:$DP32_STEP_RATE | GEMINI_ISSUES_DP4_RATE=start:$GEMINI_ISSUES_DP4_START_RATE step:$GEMINI_ISSUES_DP4_STEP_RATE | GEMINI_ISSUES_DP32_RATE=start:$GEMINI_ISSUES_DP32_START_RATE step:$GEMINI_ISSUES_DP32_STEP_RATE | STOP_THRESHOLD_MS=$STOP_THRESHOLD_MS"
log "ORDER=long_short_sp8(dp4sp8,mixed60k/deepseek_v3,rerun_from_120,bs384) -> long_short_sp8(dp4sp8,issue001/kimi_k2) -> long_short_sp8(dp4sp8,issue001/deepseek_v3) -> dp32sp1(issue001/deepseek_v3) -> long_short_sp8(dp4sp8,gemini_issues/deepseek_v3) -> dp32sp1(gemini_issues/deepseek_v3)"

run_stage "longshort" "deepseek_v3" "mixed60k" "$DEEPSEEK_MODEL" "$MIXED60K_DATASET" "long_short_sp8" "dp4sp8" "1" "$MIXED60K_RERUN_START_RATE" "$MIXED60K_RERUN_STEP_RATE" "$CHAIN_LOG_DIR/longshort_mixed60k_deepseek_v3_rerun_from120_bs384" "$MIXED60K_BATCH_SIZE"
run_stage "longshort" "kimi_k2" "issue001" "$KIMI_MODEL" "$ISSUE001_DATASET" "long_short_sp8" "dp4sp8" "1" "$DP4_START_RATE" "$DP4_STEP_RATE" "$CHAIN_LOG_DIR/longshort_issue001_kimi_k2"
run_stage "longshort" "deepseek_v3" "issue001" "$DEEPSEEK_MODEL" "$ISSUE001_DATASET" "long_short_sp8" "dp4sp8" "1" "$DP4_START_RATE" "$DP4_STEP_RATE" "$CHAIN_LOG_DIR/longshort_issue001_deepseek_v3"
run_stage "dp32" "deepseek_v3" "issue001" "$DEEPSEEK_MODEL" "$ISSUE001_DATASET" "legacy" "dp32sp1" "0" "$DP32_START_RATE" "$DP32_STEP_RATE" "$CHAIN_LOG_DIR/dp32_issue001_deepseek_v3"
run_stage "longshort" "deepseek_v3" "gemini_issues" "$DEEPSEEK_MODEL" "$GEMINI_ISSUES_DATASET" "long_short_sp8" "dp4sp8" "1" "$GEMINI_ISSUES_DP4_START_RATE" "$GEMINI_ISSUES_DP4_STEP_RATE" "$CHAIN_LOG_DIR/longshort_gemini_issues_deepseek_v3"
run_stage "dp32" "deepseek_v3" "gemini_issues" "$DEEPSEEK_MODEL" "$GEMINI_ISSUES_DATASET" "legacy" "dp32sp1" "0" "$GEMINI_ISSUES_DP32_START_RATE" "$GEMINI_ISSUES_DP32_STEP_RATE" "$CHAIN_LOG_DIR/dp32_gemini_issues_deepseek_v3"

log "ALL_DONE"
