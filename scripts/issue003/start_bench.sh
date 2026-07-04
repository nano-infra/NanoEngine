#!/bin/bash

# ================= 默认配置 (Default Configuration) =================
DEFAULT_RAY="10.102.98.166:7799"
DEFAULT_MASTER="10.102.98.166:29500"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 默认路径
DEFAULT_CSV_PATH="/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/arxiv_400k_pure_20260106_162151.csv"
DEFAULT_MODEL_PATH="/mnt/nvme1n1/ml_research/models/deepseek-v3"
DEFAULT_BASE_LOG_DIR="$ROOT_DIR/bench_logs"

# 默认并行与显存配置
DEFAULT_SEG_SIZE=65536
DEFAULT_BLOCK_SIZE=64
DEFAULT_DP=4
DEFAULT_SP=8
DEFAULT_TP=1
DEFAULT_BATCH_SIZE=192
DEFAULT_GPU_MEM=141
DEFAULT_MAX_MODEL_LEN=1000000
DEFAULT_GPU_UTIL=0.9

# 策略配置
DEFAULT_ROUTING="LeastBatch"
DEFAULT_SCHEDULER_MODE="centralized"
DEFAULT_LOOP_COUNT=16
DEFAULT_FIXED_SP_SEGMENTS=0
DEFAULT_ENABLE_DYNAMIC_SP_SIZE=1
DEFAULT_ENFORCE_EAGER=0
DEFAULT_USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER=0
DEFAULT_DYNAMIC_SP_SIZE_STRATEGY="legacy"
DEFAULT_LONG_REQUEST_SP_THRESHOLD=100000
DEFAULT_LONG_REQUEST_SP_SIZE=0
DISABLE_NON_UNIFORM_SPLIT=""  # 开关变量，非空时启用
DEFAULT_MAX_INPUT_LEN=""  # 为空表示不过滤
# ===================================================================

# 初始化变量
RAY_ADDR="$DEFAULT_RAY"
MASTER_ADDR="$DEFAULT_MASTER"
CSV_PATH="$DEFAULT_CSV_PATH"
MODEL_PATH="$DEFAULT_MODEL_PATH"
BASE_LOG_DIR="${BASE_LOG_DIR:-$DEFAULT_BASE_LOG_DIR}"

SEG_SIZE="$DEFAULT_SEG_SIZE"
DP="$DEFAULT_DP"
SP="$DEFAULT_SP"
TP="$DEFAULT_TP"
BLOCK_SIZE="$DEFAULT_BLOCK_SIZE"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
NUM_REQUESTS=""
GPU_MEM="$DEFAULT_GPU_MEM"
MAX_MODEL_LEN="$DEFAULT_MAX_MODEL_LEN"
GPU_UTIL="$DEFAULT_GPU_UTIL"
ROUTING_STRATEGY="$DEFAULT_ROUTING"
SCHEDULER_MODE="$DEFAULT_SCHEDULER_MODE"
LOOP_COUNT="$DEFAULT_LOOP_COUNT"
FIXED_SP_SEGMENTS="$DEFAULT_FIXED_SP_SEGMENTS"
ENABLE_DYNAMIC_SP_SIZE="$DEFAULT_ENABLE_DYNAMIC_SP_SIZE"
MAX_INPUT_LEN="$DEFAULT_MAX_INPUT_LEN"
ENFORCE_EAGER="$DEFAULT_ENFORCE_EAGER"
USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER="$DEFAULT_USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER"
DYNAMIC_SP_SIZE_STRATEGY="$DEFAULT_DYNAMIC_SP_SIZE_STRATEGY"
LONG_REQUEST_SP_THRESHOLD="$DEFAULT_LONG_REQUEST_SP_THRESHOLD"
LONG_REQUEST_SP_SIZE="$DEFAULT_LONG_REQUEST_SP_SIZE"

# 用于存储位置参数（Rates）
RATES=()

# 帮助函数
usage() {
    echo "Usage: $0 [options] rate1 [rate2] ..."
    echo "Options:"
    echo "  --master-addr <addr>      Master address (default: $DEFAULT_MASTER)"
    echo "  --ray-addr <addr>         Ray address (default: $DEFAULT_RAY)"
    echo "  --dataset-path <path>     Dataset CSV path"
    echo "  --model-path <path>       Model path (default: $DEFAULT_MODEL_PATH)"
    echo "  --segment-size <size>     Segment size (default: $DEFAULT_SEG_SIZE)"
    echo "  --dp-size <int>           DP (Data Parallel) size (default: $DEFAULT_DP)"
    echo "  --sp-size <int>           SP (Sequence Parallel) size (default: $DEFAULT_SP)"
    echo "  --tp-size <int>           TP (Tensor Parallel) size (default: $DEFAULT_TP)"
    echo "  --block-size <int>        KV Cache Block Size (default: $DEFAULT_BLOCK_SIZE)"
    echo "  --batch-size <int>        Batch Size / Max Num Seqs (default: $DEFAULT_BATCH_SIZE)"
    echo "  --num-requests <int>      Num requests (default: DP * SP * BatchSize)"
    echo "  --gpu-mem <int>           GPU Memory Limit GB (default: $DEFAULT_GPU_MEM)"
    echo "  --max-model-len <int>     Max Model Len (default: $DEFAULT_MAX_MODEL_LEN)"
    echo "  --gpu-util <float>        GPU Memory Utilization (default: $DEFAULT_GPU_UTIL)"
    echo "  --routing-strategy <str>  Routing Strategy (default: $DEFAULT_ROUTING)"
    echo "  --scheduler-mode <str>    Scheduler Mode (default: $DEFAULT_SCHEDULER_MODE)"
    echo "  --loop-count <int>        Loop count (default: $DEFAULT_LOOP_COUNT)"
    echo "  --fixed-sp-segments <int> Fixed SP segments (default: $DEFAULT_FIXED_SP_SEGMENTS)"
    echo "  --enable-dynamic-sp-size  Enable dynamic SP size"
    echo "  --max-input-len <int>     Filter out CSV rows with prompt_len >= this value"
    echo "  --use-new-decode-dynamic-sp-scheduler  Use the new decode dynamic SP scheduler"
    echo "  --dynamic-sp-size-strategy <str>  legacy | long_short_sp8 (default: $DEFAULT_DYNAMIC_SP_SIZE_STRATEGY)"
    echo "  --long-request-sp-threshold <int> Prompt len threshold for long_short_sp8 (default: $DEFAULT_LONG_REQUEST_SP_THRESHOLD)"
    echo "  --long-request-sp-size <int>      SP size for long requests (0 = SP size, default: $DEFAULT_LONG_REQUEST_SP_SIZE)"
    echo "  --enforce-eager           Disable cudagraph capture and enforce eager mode"
    echo "  --disable-non-uniform-split  Disable non-uniform split (flag)"
    echo "  --help                    Show this help message"
    exit 1
}

# ================= 参数解析 (Long Arguments Parsing) =================
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --master-addr)      MASTER_ADDR="$2"; shift 2 ;;
        --ray-addr)         RAY_ADDR="$2"; shift 2 ;;
        --dataset-path)     CSV_PATH="$2"; shift 2 ;;
        --model-path)       MODEL_PATH="$2"; shift 2 ;;
        --segment-size)     SEG_SIZE="$2"; shift 2 ;;
        --dp-size)          DP="$2"; shift 2 ;;
        --sp-size)          SP="$2"; shift 2 ;;
        --tp-size)          TP="$2"; shift 2 ;;
        --block-size)       BLOCK_SIZE="$2"; shift 2 ;;
        --batch-size)       BATCH_SIZE="$2"; shift 2 ;;
        --num-requests)     NUM_REQUESTS="$2"; shift 2 ;;
        --gpu-mem)          GPU_MEM="$2"; shift 2 ;;
        --max-model-len)    MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-util)         GPU_UTIL="$2"; shift 2 ;;
        --routing-strategy) ROUTING_STRATEGY="$2"; shift 2 ;;
        --scheduler-mode)   SCHEDULER_MODE="$2"; shift 2 ;;
        --loop-count)       LOOP_COUNT="$2"; shift 2 ;;
        --fixed-sp-segments) FIXED_SP_SEGMENTS="$2"; shift 2 ;;
        --enable-dynamic-sp-size) ENABLE_DYNAMIC_SP_SIZE=1; shift ;;
        --max-input-len)    MAX_INPUT_LEN="$2"; shift 2 ;;
        --use-new-decode-dynamic-sp-scheduler) USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER=1; shift ;;
        --dynamic-sp-size-strategy) DYNAMIC_SP_SIZE_STRATEGY="$2"; shift 2 ;;
        --long-request-sp-threshold) LONG_REQUEST_SP_THRESHOLD="$2"; shift 2 ;;
        --long-request-sp-size) LONG_REQUEST_SP_SIZE="$2"; shift 2 ;;
        --enforce-eager)    ENFORCE_EAGER=1; shift ;;
        --disable-non-uniform-split) DISABLE_NON_UNIFORM_SPLIT="true"; shift ;;
        --help)             usage ;;
        *)
            RATES+=("$1")
            shift
            ;;
    esac
done

# 如果未指定 num-requests，则计算默认值
if [ -z "$NUM_REQUESTS" ]; then
    NUM_REQUESTS=$((DP * SP * BATCH_SIZE))
fi

# 检查是否有 Rate 参数
if [ ${#RATES[@]} -eq 0 ]; then
    echo "Error: No rates specified."
    usage
fi

# ================= 校验逻辑 (Validation) =================
case "$ROUTING_STRATEGY" in
    RoundRobin|LeastBatch|LeastCache|WeightedScore|VLLMLoadBalance) ;;
    *) echo "Error: Invalid routing strategy '$ROUTING_STRATEGY'."; exit 1 ;;
esac

case "$SCHEDULER_MODE" in
    decentralized|centralized) ;;
    *) echo "Error: Invalid scheduler mode '$SCHEDULER_MODE'."; exit 1 ;;
esac

# ================= 准备基础信息 (Preparation) =================
DATASET_NAME=$(basename "$CSV_PATH" .csv)
MODEL_NAME=$(basename "$MODEL_PATH")
PROGRESS_LOG="$BASE_LOG_DIR/run_progress.log"
EP=$((DP * SP * TP))

log_progress() {
    local msg="$1"
    local now=$(TZ='Asia/Shanghai' date "+%Y-%m-%d %H:%M:%S")
    mkdir -p "$(dirname "$PROGRESS_LOG")"
    echo "[$now] $msg" >> "$PROGRESS_LOG"
}

# 处理利用率字符串用于路径 (去掉小数点)
MEM_TAG="MEM$(echo "$GPU_UTIL" | sed 's/0\.//')"

# max-input-len 标识
if [[ -n "$MAX_INPUT_LEN" ]]; then
    INPUT_LEN_TAG="_maxin${MAX_INPUT_LEN}"
else
    INPUT_LEN_TAG=""
fi

echo "================================================"
echo "Model Name  : $MODEL_NAME"
echo "Dataset     : $DATASET_NAME"
echo "Strategy    : DP=$DP, SP=$SP, EP=$EP, TP=$TP, BK_SZ=$BLOCK_SIZE"
echo "Routing     : $ROUTING_STRATEGY"
echo "SchedMod    : $SCHEDULER_MODE"
echo "LBCandRatio : $LEASTBATCH_TOKEN_CANDIDATE_RATIO"
echo "BatchSz     : $BATCH_SIZE"
echo "GPU Util    : $GPU_UTIL ($MEM_TAG)"
echo "GPU Mem     : ${GPU_MEM}GB"
echo "Max Len     : $MAX_MODEL_LEN"
echo "Max Input   : ${MAX_INPUT_LEN:-unlimited}"
echo "Fixed SP Seg: $FIXED_SP_SEGMENTS"
echo "Enable Dynamic SP Size: $ENABLE_DYNAMIC_SP_SIZE"
echo "New Decode Dynamic SP Scheduler: $USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER"
echo "Dynamic SP Size Strategy: $DYNAMIC_SP_SIZE_STRATEGY"
echo "Long Request SP Threshold: $LONG_REQUEST_SP_THRESHOLD"
echo "Enforce Eager: $ENFORCE_EAGER"
echo "Model Path  : $MODEL_PATH"
echo "Rates       : ${RATES[*]}"
echo "================================================"

log_progress "=== NEW BATCH STARTED ==="
log_progress "Model: $MODEL_NAME | Dataset: $DATASET_NAME"
log_progress "Parallel: DP=$DP, SP=$SP, EP=$EP, TP=$TP | Scheduler: $SCHEDULER_MODE"
log_progress "SegSize=$SEG_SIZE | BatchSize=$BATCH_SIZE | MaxLen=$MAX_MODEL_LEN | MaxInput=${MAX_INPUT_LEN:-unlimited} | FixedSP=$FIXED_SP_SEGMENTS"
log_progress "EnableDynamicSPSize=$ENABLE_DYNAMIC_SP_SIZE"
log_progress "UseNewDecodeDynamicSPScheduler=$USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER"
log_progress "DynamicSPSizeStrategy=$DYNAMIC_SP_SIZE_STRATEGY"
log_progress "LongRequestSPThreshold=$LONG_REQUEST_SP_THRESHOLD"
log_progress "EnforceEager=$ENFORCE_EAGER"
log_progress "GPU: ${GPU_MEM}GB, Util=$GPU_UTIL | Routing=$ROUTING_STRATEGY | Loop=$LOOP_COUNT"
log_progress "LeastBatchTokenCandidateRatio=$LEASTBATCH_TOKEN_CANDIDATE_RATIO"
log_progress "NumReqs=$NUM_REQUESTS | Rates=${RATES[*]}"

# ================= Python 脚本路径 =================
PYTHON_SCRIPT="$SCRIPT_DIR/bench_serving_overhead.py"
BUILD_LIB_DIR="$ROOT_DIR/build/lib"
if [[ -d "$BUILD_LIB_DIR" ]]; then
    export PYTHONPATH="$ROOT_DIR:$BUILD_LIB_DIR${PYTHONPATH:+:$PYTHONPATH}"
else
    export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

# ================= 执行循环 (Execution Loop) =================
for rate in "${RATES[@]}"; do
    TIMESTAMP=$(TZ='Asia/Shanghai' date "+%Y%m%d_%H%M%S")

    # 策略标识字符串 (缩写)
    # 路由缩写: LeastBatch->LB, LeastCache->LC, RoundRobin->RR, VLLMLoadBalance->VLLM
    rt_short=""
    case "$ROUTING_STRATEGY" in
        LeastBatch)      rt_short="LB" ;;
        LeastCache)      rt_short="LC" ;;
        RoundRobin)      rt_short="RR" ;;
        VLLMLoadBalance) rt_short="VLLM" ;;
        *)               rt_short="$ROUTING_STRATEGY" ;;
    esac
    # 调度缩写
    sc_short=""
    case "$SCHEDULER_MODE" in
        centralized)   sc_short="cen" ;;
        decentralized) sc_short="dec" ;;
        *)             sc_short="$SCHEDULER_MODE" ;;
    esac
    # segment size 缩写 (65536->64k)
    seg_short=$(( SEG_SIZE / 1024 ))k
    # max-input-len 缩写
    maxin_tag=""
    if [[ -n "$MAX_INPUT_LEN" ]]; then
        maxin_tag="_maxin$(( MAX_INPUT_LEN / 1000 ))k"
    fi
    # 可选标签
    extra_tags=""
    if [[ -n "$DISABLE_NON_UNIFORM_SPLIT" ]]; then
        extra_tags="${extra_tags}_uni"
    fi
    if [[ "$FIXED_SP_SEGMENTS" -ne 0 ]]; then
        extra_tags="${extra_tags}_fsp${FIXED_SP_SEGMENTS}"
    fi
    if [[ "$DYNAMIC_SP_SIZE_STRATEGY" != "legacy" ]]; then
        extra_tags="${extra_tags}_${DYNAMIC_SP_SIZE_STRATEGY}_thr${LONG_REQUEST_SP_THRESHOLD}"
        if [[ "$LONG_REQUEST_SP_SIZE" -ne 0 ]]; then
            extra_tags="${extra_tags}_sp${LONG_REQUEST_SP_SIZE}"
        fi
    fi
    STRATEGY_STR="dp${DP}sp${SP}_seg${seg_short}_n${NUM_REQUESTS}_r${rate}_bs${BATCH_SIZE}_${rt_short}_${sc_short}${maxin_tag}${extra_tags}"

    CURRENT_LOG_DIR="$BASE_LOG_DIR/$MODEL_NAME/$DATASET_NAME/$STRATEGY_STR"
    mkdir -p "$CURRENT_LOG_DIR"

    LOG_FILE="$CURRENT_LOG_DIR/${TIMESTAMP}.log"
    JSON_FILE="$CURRENT_LOG_DIR/${TIMESTAMP}.json"

    # 构建 Python 命令
    CMD=(
        python "$PYTHON_SCRIPT"
        --dataset csv
        --csv-path "$CSV_PATH"
        --num-requests "$NUM_REQUESTS"
        --request-rate "$rate"
        --sp "$SP"
        --dp "$DP"
        --ep "$EP"
        --tp "$TP"
        --max-num-seqs "$BATCH_SIZE"
        --gpu-memory-limit-gb "$GPU_MEM"
        --gpu-memory-utilization "$GPU_UTIL"
        --max-model-len "$MAX_MODEL_LEN"
        --dummy-prefill
        --ray-address "$RAY_ADDR"
        --master-address "$MASTER_ADDR"
        --loop-count "$LOOP_COUNT"
        --model-path "$MODEL_PATH"
        --routing-strategy "$ROUTING_STRATEGY"
        --itl-log-path "$JSON_FILE"
        --segment-size "$SEG_SIZE"
        --scheduler-mode "$SCHEDULER_MODE"
        --fixed-sp-segments "$FIXED_SP_SEGMENTS"
        --dynamic-sp-size-strategy "$DYNAMIC_SP_SIZE_STRATEGY"
        --long-request-sp-threshold "$LONG_REQUEST_SP_THRESHOLD"
        --long-request-sp-size "$LONG_REQUEST_SP_SIZE"
    )

    # 如果启用了 disable_non_uniform_split
    if [[ -n "$DISABLE_NON_UNIFORM_SPLIT" ]]; then
        CMD+=(--disable-non-uniform-split)
    fi

    if [[ "$ENFORCE_EAGER" -ne 0 ]]; then
        CMD+=(--enforce-eager)
    fi
    if [[ "$USE_NEW_DECODE_DYNAMIC_SP_SCHEDULER" -ne 0 ]]; then
        CMD+=(--use-new-decode-dynamic-sp-scheduler)
    fi
    if [[ "$ENABLE_DYNAMIC_SP_SIZE" -ne 0 ]]; then
        CMD+=(--enable-dynamic-sp-size)
    fi

    # 如果设置了 max-input-len
    if [[ -n "$MAX_INPUT_LEN" ]]; then
        CMD+=(--max-input-len "$MAX_INPUT_LEN")
    fi

    # 记录元数据
    {
        echo "================= Benchmark Metadata ================="
        echo "Time (Beijing): $TIMESTAMP"
        echo "Model Name: $MODEL_NAME"
        echo "Model Path: $MODEL_PATH"
        echo "Dataset: $DATASET_NAME"
        echo "Max Input Len: ${MAX_INPUT_LEN:-unlimited}"
        echo "Settings: $STRATEGY_STR"
        echo ""
        echo "================= Reproduce Command ================="
        echo "RAY_DEDUP_LOGS=0 ${CMD[*]}"
        echo "====================================================="
        echo ""
    } > "$LOG_FILE"

    echo ">>> Starting Rate: $rate | Output: $CURRENT_LOG_DIR"

    # 执行命令并捕获退出码
    cd "$ROOT_DIR"
    set -o pipefail
    RAY_DEDUP_LOGS=0 "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=$?
    set +o pipefail

    if [ $EXIT_CODE -eq 0 ]; then
        log_progress " <- Finished Rate: $rate. Status: SUCCESS"
    else
        log_progress " <- Finished Rate: $rate. Status: FAILED (Code $EXIT_CODE)"
    fi

    sleep 15
done

log_progress "=== BATCH FINISHED ==="
