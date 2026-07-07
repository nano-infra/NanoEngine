#!/bin/bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

# Do not disable TorchDynamo from this benchmark script. If the parent shell
# or a previous run exported TORCHDYNAMO_DISABLE=1, avoid passing it to the
# benchmark driver process.
unset TORCHDYNAMO_DISABLE

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SWEEP_SCRIPT="$ROOT_DIR/scripts/issue003/run_issue003_deepseek_rate5_sweep_0409_linbinbin.sh"

RAY_ADDR="${RAY_ADDR:-10.102.97.183:7822}"
MASTER_ADDR="${MASTER_ADDR:-10.102.97.183:29522}"

DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-/mnt/nvme1n1/ml_research/models/deepseek-v3}"
KIMI_MODEL="${KIMI_MODEL:-/mnt/nvme1n1/ml_research/models/models--moonshotai--Kimi-K2-Instruct-0905/snapshots/7152993552508c9f22042b3bb93b5e6acd06ce73}"

ISSUE003_DATASET="${ISSUE003_DATASET:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.03_n60000.csv}"
ISSUE005_DATASET="${ISSUE005_DATASET:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.05_n60000_60k.csv}"
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
SP_BACKEND="${SP_BACKEND:-hao_basic}"
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
GEMINI_ISSUES_DP4_MAX_RATE="${GEMINI_ISSUES_DP4_MAX_RATE:-8}"
GEMINI_ISSUES_DP32_START_RATE="${GEMINI_ISSUES_DP32_START_RATE:-$DP32_START_RATE}"
GEMINI_ISSUES_DP32_STEP_RATE="${GEMINI_ISSUES_DP32_STEP_RATE:-2}"
GEMINI_ISSUES_DPSK_DP4_START_RATE="${GEMINI_ISSUES_DPSK_DP4_START_RATE:-2}"
GEMINI_ISSUES_DPSK_DP4_STEP_RATE="${GEMINI_ISSUES_DPSK_DP4_STEP_RATE:-2}"
GEMINI_ISSUES_DPSK_DP4_MAX_RATE="${GEMINI_ISSUES_DPSK_DP4_MAX_RATE:-8}"
ISSUE005_DPSK_DP4_START_RATE="${ISSUE005_DPSK_DP4_START_RATE:-45}"
ISSUE005_DPSK_DP4_STEP_RATE="${ISSUE005_DPSK_DP4_STEP_RATE:--5}"
ISSUE005_DPSK_DP4_MAX_RATE="${ISSUE005_DPSK_DP4_MAX_RATE:-15}"
DP4_R20_200_START_RATE="${DP4_R20_200_START_RATE:-20}"
DP4_R20_200_STEP_RATE="${DP4_R20_200_STEP_RATE:-20}"
DP4_R20_200_MAX_RATE="${DP4_R20_200_MAX_RATE:-200}"
DP4_R10_190_START_RATE="${DP4_R10_190_START_RATE:-10}"
DP4_R10_190_STEP_RATE="${DP4_R10_190_STEP_RATE:-10}"
DP4_R10_190_MAX_RATE="${DP4_R10_190_MAX_RATE:-190}"
DP4_R80_200_START_RATE="${DP4_R80_200_START_RATE:-80}"
DP4_R80_200_STEP_RATE="${DP4_R80_200_STEP_RATE:-5}"
DP4_R80_200_MAX_RATE="${DP4_R80_200_MAX_RATE:-200}"
MAX_RATE="${MAX_RATE:-500}"
STOP_THRESHOLD_MS="${STOP_THRESHOLD_MS:-100}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-20}"
MAX_RETRIES="${MAX_RETRIES:-5}"
NANODEPLOY_LOG_DECODE_A2A_MASKS="${NANODEPLOY_LOG_DECODE_A2A_MASKS:-0}"
NANODEPLOY_LOG_DECODE_STEP_DETAIL="${NANODEPLOY_LOG_DECODE_STEP_DETAIL:-0}"
LONG_REQUEST_SP_THRESHOLD="${LONG_REQUEST_SP_THRESHOLD:-100000}"
LONG_REQUEST_SP_SIZE="${LONG_REQUEST_SP_SIZE:-8}"
REUSE_SWEEP_LOG_ROOT="${REUSE_SWEEP_LOG_ROOT:-$ROOT_DIR/bench_logs/issue001_issue003_mixed60k_longshort_dp4dp32_bs256_20260408_165526}"
GEMINI_ISSUES_DPSK_DP4_LOG_DIR="${GEMINI_ISSUES_DPSK_DP4_LOG_DIR:-$REUSE_SWEEP_LOG_ROOT/longshort_gemini_issues_deepseek_v3_r2_8}"
ISSUE005_DPSK_DP4_LOG_DIR="${ISSUE005_DPSK_DP4_LOG_DIR:-$REUSE_SWEEP_LOG_ROOT/longshort_issue005_random_deepseek_v3_r10_190}"

NODE_PLAN="${NODE_PLAN:-10.102.97.183,10.102.98.166,10.102.98.154,10.102.252.174}"
RUN_TAG="${RUN_TAG:-issue005_rate40_dp32_dp4sp8_longsp8_4node_original_setting_$(date -u +%Y%m%d_%H%M%S)}"
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
    local max_rate="${13:-$MAX_RATE}"
    local routing_strategy="${14:-LeastBatch}"
    local stage_long_request_sp_size="${15:-$LONG_REQUEST_SP_SIZE}"

    log "START phase=$phase model=$model_key dataset=$dataset_key strategy=$strategy sweep=$sweep_strategy routing=$routing_strategy dynamic_sp=$enable_dynamic_sp_size long_sp_size=$stage_long_request_sp_size start_rate=$start_rate step_rate=$step_rate max_rate=$max_rate batch_size=$stage_batch_size"

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
        SP_BACKEND="$SP_BACKEND" \
        ENFORCE_EAGER="$ENFORCE_EAGER" \
        START_RATE="$start_rate" \
        STEP_RATE="$step_rate" \
        MAX_RATE="$max_rate" \
        STOP_THRESHOLD_MS="$STOP_THRESHOLD_MS" \
        SLEEP_BETWEEN_RUNS="$SLEEP_BETWEEN_RUNS" \
        MAX_RETRIES="$MAX_RETRIES" \
        NANODEPLOY_LOG_DECODE_A2A_MASKS="$NANODEPLOY_LOG_DECODE_A2A_MASKS" \
        NANODEPLOY_LOG_DECODE_STEP_DETAIL="$NANODEPLOY_LOG_DECODE_STEP_DETAIL" \
        MODEL_PATH="$model_path" \
        DATASET_PATH="$dataset_path" \
        BASE_LOG_DIR="$base_log_dir" \
        RUN_TAG="$(basename "$base_log_dir")" \
        ENABLE_DYNAMIC_SP_SIZE="$enable_dynamic_sp_size" \
        USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER=0 \
        DYNAMIC_SP_SIZE_STRATEGY="$strategy" \
        LONG_REQUEST_SP_THRESHOLD="$LONG_REQUEST_SP_THRESHOLD" \
        LONG_REQUEST_SP_SIZE="$stage_long_request_sp_size" \
        SWEEP_STRATEGY="$sweep_strategy" \
        SWEEP_ROUTING="$routing_strategy" \
        bash "$SWEEP_SCRIPT"

    record_stage "$phase" "$model_key" "$dataset_key" "$base_log_dir" "done"
    log "DONE phase=$phase model=$model_key dataset=$dataset_key strategy=$strategy sweep=$sweep_strategy routing=$routing_strategy dynamic_sp=$enable_dynamic_sp_size long_sp_size=$stage_long_request_sp_size start_rate=$start_rate step_rate=$step_rate max_rate=$max_rate batch_size=$stage_batch_size"
}

ensure_headers

log "CHAIN_TAG=$RUN_TAG"
log "RAY_ADDR=$RAY_ADDR MASTER_ADDR=$MASTER_ADDR"
log "NODE_PLAN=$NODE_PLAN"
log "SP_BACKEND=$SP_BACKEND"
# log "BATCH_SIZE(plan)=$BATCH_SIZE GPU_UTIL=$GPU_UTIL ENFORCE_EAGER=$ENFORCE_EAGER"
# log "RATE_PLAN mixed60k=90 | issue001=35 | issue003_random=2.5,22.5,25 | issue005_random=2.5,17.5 | gemini_issues=1.75 | kimi_issue005_random_dp4=5,10,15,20,25,30,35,40 | kimi_issue005_random_dp32=5,10,15,20,25,30,35,40 | kimi_issue003_random_dp4=5,10,15,20,25,30,35,40 | kimi_issue003_random_dp32=5,10,15,20,25,30,35,40 | STOP_THRESHOLD_MS=$STOP_THRESHOLD_MS"
# log "ORDER=long_short(dp4sp8,mixed60k/deepseek_v3,r90) -> long_short(dp4sp8,issue001/deepseek_v3,r35) -> long_short(dp4sp8,issue003_random/deepseek_v3,r2.5,22.5,25) -> long_short(dp4sp8,issue005_random/deepseek_v3,r2.5,17.5) -> long_short(dp4sp8,gemini_issues/deepseek_v3,r1.75) -> long_short(dp4sp8,issue005_random/kimi_k2,r5,10,15,20,25,30,35,40) -> dp32sp1(issue005_random/kimi_k2,r5,10,15,20,25,30,35,40) -> long_short(dp4sp8,issue003_random/kimi_k2,r5,10,15,20,25,30,35,40) -> dp32sp1(issue003_random/kimi_k2,r5,10,15,20,25,30,35,40)"

##### 全短
# for rate in 90; do
#     rate_tag="${rate//./p}"
#     run_stage "longshort" "deepseek_v3" "mixed60k" "$DEEPSEEK_MODEL" "$MIXED60K_DATASET" "long_short" "dp4sp8" "1" "$rate" "1" "$CHAIN_LOG_DIR/longshort_mixed60k_deepseek_v3_r${rate_tag}" "$BATCH_SIZE" "$rate"
# done

##### 1%
# for rate in 35; do
#     rate_tag="${rate//./p}"
#     run_stage "longshort" "deepseek_v3" "issue001" "$DEEPSEEK_MODEL" "$ISSUE001_DATASET" "long_short" "dp4sp8" "1" "$rate" "1" "$CHAIN_LOG_DIR/longshort_issue001_deepseek_v3_r${rate_tag}" "$BATCH_SIZE" "$rate"
# done

##### issue005 DPSK long_short SP size sweep
for long_sp_size in 8; do
# for long_sp_size in 2 4 8; do
    for rate in 40; do
        rate_tag="${rate//./p}"
        run_stage \
            "longshort" \
            "deepseek_v3" \
            "issue005_random" \
            "$DEEPSEEK_MODEL" \
            "$ISSUE005_DATASET" \
            "long_short_sp8" \
            "dp4sp8" \
            "1" \
            "$rate" \
            "1" \
            "$CHAIN_LOG_DIR/longshort_issue005_random_deepseek_v3_sp${long_sp_size}_r${rate_tag}" \
            "$BATCH_SIZE" \
            "$rate" \
            "LeastBatch" \
            "$long_sp_size"
    done
done

##### DPSK issue005 dp32 LeastBatch rate 40
# Skipped for this run: user requested only the SP-enabled dp4sp8 setting.
# for rate in 40; do
#     rate_tag="${rate//./p}"
#     run_stage "dp32" "deepseek_v3" "issue005_random" "$DEEPSEEK_MODEL" "$ISSUE005_DATASET" "legacy" "dp32sp1" "0" "$rate" "1" "$CHAIN_LOG_DIR/dp32_issue005_random_deepseek_v3_LB_r${rate_tag}" "$BATCH_SIZE" "$rate" "LeastBatch" "1"
# done

##### 3%
# for rate in 2.5 22.5 25; do
#     rate_tag="${rate//./p}"
#     run_stage "longshort" "deepseek_v3" "issue003_random" "$DEEPSEEK_MODEL" "$ISSUE003_DATASET" "long_short" "dp4sp8" "1" "$rate" "1" "$CHAIN_LOG_DIR/longshort_issue003_random_deepseek_v3_r${rate_tag}" "$BATCH_SIZE" "$rate"
# done

##### 5% dp4sp8
# for rate in 2.5 17.5; do
#     rate_tag="${rate//./p}"
#     run_stage "longshort" "deepseek_v3" "issue005_random" "$DEEPSEEK_MODEL" "$ISSUE005_DATASET" "long_short" "dp4sp8" "1" "$rate" "1" "$CHAIN_LOG_DIR/longshort_issue005_random_deepseek_v3_r${rate_tag}" "$BATCH_SIZE" "$rate"
# done

##### 5% dp32 LeastBatch
# for rate in 30 40 45; do
#     rate_tag="${rate//./p}"
#     run_stage "dp32" "deepseek_v3" "issue005_random" "$DEEPSEEK_MODEL" "$ISSUE005_DATASET" "legacy" "dp32sp1" "0" "$rate" "1" "$CHAIN_LOG_DIR/dp32_issue005_random_deepseek_v3_LB_r${rate_tag}" "$BATCH_SIZE" "$rate" "LeastBatch"
# done

##### 全长
# for rate in 1.75; do
#     rate_tag="${rate//./p}"
#     run_stage "longshort" "deepseek_v3" "gemini_issues" "$DEEPSEEK_MODEL" "$GEMINI_ISSUES_DATASET" "long_short" "dp4sp8" "1" "$rate" "1" "$CHAIN_LOG_DIR/longshort_gemini_issues_deepseek_v3_r${rate_tag}" "$BATCH_SIZE" "$rate"
# done

##### KIMI 5%
# for rate in  50 ; do
#     rate_tag="${rate//./p}"
#     run_stage "longshort" "kimi_k2" "issue005_random" "$KIMI_MODEL" "$ISSUE005_DATASET" "long_short" "dp4sp8" "1" "$rate" "1" "$CHAIN_LOG_DIR/longshort_issue005_random_kimi_k2_r${rate_tag}" "$BATCH_SIZE" "$rate"
# done

##### KIMI 5% dp32
# for rate in 2.5 ; do
#     rate_tag="${rate//./p}"
#     run_stage "dp32" "kimi_k2" "issue005_random" "$KIMI_MODEL" "$ISSUE005_DATASET" "legacy" "dp32sp1" "0" "$rate" "1" "$CHAIN_LOG_DIR/dp32_issue005_random_kimi_k2_r${rate_tag}" "$BATCH_SIZE" "$rate"
# done

##### KIMI 3%
# for rate in 2.5; do
#     rate_tag="${rate//./p}"
#     run_stage "longshort" "kimi_k2" "issue003_random" "$KIMI_MODEL" "$ISSUE003_DATASET" "long_short" "dp4sp8" "1" "$rate" "1" "$CHAIN_LOG_DIR/longshort_issue003_random_kimi_k2_r${rate_tag}" "$BATCH_SIZE" "$rate"
# done

##### KIMI 3% dp32
# for rate in   2.5; do
#     rate_tag="${rate//./p}"
#     run_stage "dp32" "kimi_k2" "issue003_random" "$KIMI_MODEL" "$ISSUE003_DATASET" "legacy" "dp32sp1" "0" "$rate" "1" "$CHAIN_LOG_DIR/dp32_issue003_random_kimi_k2_r${rate_tag}" "$BATCH_SIZE" "$rate"
# done

##### DPSK issue001 dp32 LeastBatch / LeastCache
# log "APPEND_PLAN dpsk_issue001_dp32 routing=LeastBatch,LeastCache rates=60,70"
# for routing in LeastBatch LeastCache; do
#     routing_tag="$(case "$routing" in LeastBatch) echo LB ;; LeastCache) echo LC ;; *) echo "$routing" ;; esac)"
#     for rate in 60 70; do
#         rate_tag="${rate//./p}"
#         run_stage "dp32" "deepseek_v3" "issue001" "$DEEPSEEK_MODEL" "$ISSUE001_DATASET" "legacy" "dp32sp1" "0" "$rate" "1" "$CHAIN_LOG_DIR/dp32_issue001_deepseek_v3_${routing_tag}_r${rate_tag}" "$BATCH_SIZE" "$rate" "$routing"
#     done
# done

log "ALL_DONE"
