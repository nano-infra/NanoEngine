#!/bin/bash

# ================= 默认配置 (Default Configuration) =================
DEFAULT_RAY="10.102.252.174:6379"
DEFAULT_MASTER="10.102.252.174:29500"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 默认路径
DEFAULT_CSV_PATH="/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/arxiv_400k_pure_20260106_162151.csv"
DEFAULT_MODEL_PATH="/mnt/nvme1n1/ml_research/chenjiefei/models/deepseek-v3"
DEFAULT_BASE_LOG_DIR="$ROOT_DIR/bench_logs"

# 默认并行与显存配置
DEFAULT_SEG_SIZE=65536
DEFAULT_BLOCK_SIZE=64
DEFAULT_DP=4
DEFAULT_SP=8
DEFAULT_TP=1
DEFAULT_BATCH_SIZE=256
DEFAULT_GPU_MEM=141
DEFAULT_MAX_MODEL_LEN=1000000
DEFAULT_GPU_UTIL=0.9

# 策略配置
DEFAULT_ROUTING="LeastBatch"
DEFAULT_SCHEDULER_ARCH="legacy_global"
DEFAULT_ROUTER_POLICY="least_batch"
DEFAULT_SP_MASTER_SELECTOR="LeastBatch"
DEFAULT_LOOP_COUNT=16
DEFAULT_FIXED_SP_SIZE=0
DEFAULT_SP_BACKEND="hao_basic"
DEFAULT_CUDA_GRAPH_MODE="full"
DEFAULT_ENFORCE_EAGER=0
DEFAULT_DYNAMIC_SP_SIZE_STRATEGY="legacy"
DEFAULT_DYNAMIC_SP_BUCKET_PRESET="none"
DEFAULT_DIAGNOSTIC_LOG_INTERVAL=0
DEFAULT_SLOW_ADD_THRESHOLD_MS=0
DEFAULT_QUANTUM_DIAGNOSTICS=0
DISABLE_NON_UNIFORM_SPLIT=""  # 开关变量，非空时启用
DEFAULT_MAX_INPUT_LEN=""  # 为空表示不过滤
DEFAULT_MAX_REQUEST_TOKENS=910000
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
SCHEDULER_ARCH="$DEFAULT_SCHEDULER_ARCH"
ROUTER_POLICY="${ROUTER_POLICY:-$DEFAULT_ROUTER_POLICY}"
SP_MASTER_SELECTOR="${SP_MASTER_SELECTOR:-$DEFAULT_SP_MASTER_SELECTOR}"
LOOP_COUNT="$DEFAULT_LOOP_COUNT"
FIXED_SP_SIZE="${FIXED_SP_SIZE:-$DEFAULT_FIXED_SP_SIZE}"
SP_BACKEND="${SP_BACKEND:-$DEFAULT_SP_BACKEND}"
CUDA_GRAPH_MODE="${CUDA_GRAPH_MODE:-$DEFAULT_CUDA_GRAPH_MODE}"
MAX_INPUT_LEN="$DEFAULT_MAX_INPUT_LEN"
MAX_REQUEST_TOKENS="${MAX_REQUEST_TOKENS:-$DEFAULT_MAX_REQUEST_TOKENS}"
ENFORCE_EAGER="$DEFAULT_ENFORCE_EAGER"
DYNAMIC_SP_SIZE_STRATEGY="$DEFAULT_DYNAMIC_SP_SIZE_STRATEGY"
DYNAMIC_SP_BUCKET_PRESET="$DEFAULT_DYNAMIC_SP_BUCKET_PRESET"
DIAGNOSTIC_LOG_INTERVAL="${DIAGNOSTIC_LOG_INTERVAL:-$DEFAULT_DIAGNOSTIC_LOG_INTERVAL}"
SLOW_ADD_THRESHOLD_MS="${SLOW_ADD_THRESHOLD_MS:-$DEFAULT_SLOW_ADD_THRESHOLD_MS}"
HIERARCHICAL_EXECUTION_TRACE="${HIERARCHICAL_EXECUTION_TRACE:-0}"
QUANTUM_DIAGNOSTICS="${QUANTUM_DIAGNOSTICS:-${HIERARCHICAL_QUANTUM_DIAGNOSTICS:-$DEFAULT_QUANTUM_DIAGNOSTICS}}"
RUN_LABEL="${RUN_LABEL:-}"

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
    echo "  --scheduler-arch <str>    Scheduler architecture (default: $DEFAULT_SCHEDULER_ARCH)"
    echo "  --router-policy <str>     round_robin | least_batch | least_cache (default: $DEFAULT_ROUTER_POLICY)"
    echo "  --sp-master-selector <str> RoundRobin | LeastBatch | LeastCache (default: $DEFAULT_SP_MASTER_SELECTOR)"
    echo "  --loop-count <int>        Loop count (default: $DEFAULT_LOOP_COUNT)"
    echo "  --fixed-sp-size <int>     Fixed SP size baseline (0 = disabled, default: $DEFAULT_FIXED_SP_SIZE)"
    echo "  --sp-backend <str>        hao_basic | nccl (default: $DEFAULT_SP_BACKEND)"
    echo "  --cuda-graph-mode <str>   full | piecewise (default: $DEFAULT_CUDA_GRAPH_MODE)"
    echo "  --max-input-len <int>     Filter out CSV rows with prompt_len >= this value"
    echo "  --max-request-tokens <int> Filter out CSV rows with prompt_len + output_len above this value (default: $DEFAULT_MAX_REQUEST_TOKENS; 0 disables)"
    echo "  --dynamic-sp-size-strategy <str>  legacy | bucket (default: $DEFAULT_DYNAMIC_SP_SIZE_STRATEGY)"
    echo "  --dynamic-sp-bucket-preset <str>  none | deepseek_v3 | kimi_k2 (default: $DEFAULT_DYNAMIC_SP_BUCKET_PRESET)"
    echo "  --diagnostic-log-interval <sec>   Structured client/scheduler snapshot interval (0 = disabled)"
    echo "  --slow-add-threshold-ms <ms>      Log slow async due-batch submissions (0 = disabled)"
    echo "  --hierarchical-execution-trace    Capture high-overhead per-rank hierarchical traces"
    echo "  --quantum-diagnostics     Capture lightweight decode quantum timing/load JSONL"
    echo "  --hierarchical-quantum-diagnostics Deprecated alias for --quantum-diagnostics"
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
        --scheduler-arch)   SCHEDULER_ARCH="$2"; shift 2 ;;
        --router-policy)    ROUTER_POLICY="$2"; shift 2 ;;
        --sp-master-selector) SP_MASTER_SELECTOR="$2"; shift 2 ;;
        --loop-count)       LOOP_COUNT="$2"; shift 2 ;;
        --fixed-sp-size)     FIXED_SP_SIZE="$2"; shift 2 ;;
        --sp-backend)       SP_BACKEND="$2"; shift 2 ;;
        --cuda-graph-mode)  CUDA_GRAPH_MODE="$2"; shift 2 ;;
        --run-label)        RUN_LABEL="$2"; shift 2 ;;
        --max-input-len)    MAX_INPUT_LEN="$2"; shift 2 ;;
        --max-request-tokens) MAX_REQUEST_TOKENS="$2"; shift 2 ;;
        --dynamic-sp-size-strategy) DYNAMIC_SP_SIZE_STRATEGY="$2"; shift 2 ;;
        --dynamic-sp-bucket-preset) DYNAMIC_SP_BUCKET_PRESET="$2"; shift 2 ;;
        --diagnostic-log-interval) DIAGNOSTIC_LOG_INTERVAL="$2"; shift 2 ;;
        --slow-add-threshold-ms) SLOW_ADD_THRESHOLD_MS="$2"; shift 2 ;;
        --hierarchical-execution-trace) HIERARCHICAL_EXECUTION_TRACE=1; shift ;;
        --quantum-diagnostics) QUANTUM_DIAGNOSTICS=1; shift ;;
        --hierarchical-quantum-diagnostics) QUANTUM_DIAGNOSTICS=1; shift ;;
        --enforce-eager)    ENFORCE_EAGER=1; shift ;;
        --disable-non-uniform-split) DISABLE_NON_UNIFORM_SPLIT="true"; shift ;;
        --help)             usage ;;
        --*)
            echo "Error: Unknown option '$1'." >&2
            exit 1
            ;;
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
    RoundRobin|LeastBatch|LeastCache) ;;
    *) echo "Error: Invalid routing strategy '$ROUTING_STRATEGY'."; exit 1 ;;
esac

case "$SCHEDULER_ARCH" in
    legacy_global|hierarchical) ;;
    *) echo "Error: Invalid scheduler architecture '$SCHEDULER_ARCH'."; exit 1 ;;
esac

case "$ROUTER_POLICY" in
    round_robin|least_batch|least_cache) ;;
    *) echo "Error: Invalid router policy '$ROUTER_POLICY'."; exit 1 ;;
esac

case "$SP_MASTER_SELECTOR" in
    RoundRobin|LeastBatch|LeastCache) ;;
    *) echo "Error: Invalid SP master selector '$SP_MASTER_SELECTOR'."; exit 1 ;;
esac

case "$SP_BACKEND" in
    hao_basic|nccl) ;;
    *) echo "Error: Invalid SP backend '$SP_BACKEND'."; exit 1 ;;
esac

case "$CUDA_GRAPH_MODE" in
    full|piecewise) ;;
    *) echo "Error: Invalid CUDA graph mode '$CUDA_GRAPH_MODE'."; exit 1 ;;
esac

case "$DYNAMIC_SP_SIZE_STRATEGY" in
    legacy|bucket) ;;
    *) echo "Error: Invalid dynamic SP size strategy '$DYNAMIC_SP_SIZE_STRATEGY'."; exit 1 ;;
esac

case "$DYNAMIC_SP_BUCKET_PRESET" in
    none|deepseek_v3|kimi_k2) ;;
    *) echo "Error: Invalid dynamic SP bucket preset '$DYNAMIC_SP_BUCKET_PRESET'."; exit 1 ;;
esac

if [[ "$DYNAMIC_SP_SIZE_STRATEGY" == "bucket" && "$DYNAMIC_SP_BUCKET_PRESET" == "none" ]]; then
    echo "Error: bucket strategy requires --dynamic-sp-bucket-preset."
    exit 1
fi
if [[ "$DYNAMIC_SP_SIZE_STRATEGY" != "bucket" && "$DYNAMIC_SP_BUCKET_PRESET" != "none" ]]; then
    echo "Error: --dynamic-sp-bucket-preset requires bucket strategy."
    exit 1
fi

if [[ "$HIERARCHICAL_EXECUTION_TRACE" -ne 0 ]]; then
    if [[ "$SCHEDULER_ARCH" != "hierarchical" ]]; then
        echo "Error: --hierarchical-execution-trace requires --scheduler-arch hierarchical."
        exit 1
    fi
    case "$DIAGNOSTIC_LOG_INTERVAL" in
        0|0.0|0.00)
            echo "Error: --hierarchical-execution-trace requires a positive --diagnostic-log-interval."
            exit 1
            ;;
    esac
fi

if ! [[ "$QUANTUM_DIAGNOSTICS" =~ ^[01]$ ]]; then
    echo "Error: QUANTUM_DIAGNOSTICS must be 0 or 1."
    exit 1
fi

if ! [[ "$MAX_REQUEST_TOKENS" =~ ^[0-9]+$ ]]; then
    echo "Error: --max-request-tokens must be a non-negative integer."
    exit 1
fi

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
echo "SchedArch   : $SCHEDULER_ARCH"
echo "RouterPolicy: $ROUTER_POLICY"
echo "SPMasterSel : $SP_MASTER_SELECTOR"
echo "LBCandRatio : $LEASTBATCH_TOKEN_CANDIDATE_RATIO"
echo "BatchSz     : $BATCH_SIZE"
echo "GPU Util    : $GPU_UTIL ($MEM_TAG)"
echo "GPU Mem     : ${GPU_MEM}GB"
echo "Max Len     : $MAX_MODEL_LEN"
echo "Max Input   : ${MAX_INPUT_LEN:-unlimited}"
echo "Max Request : ${MAX_REQUEST_TOKENS:-unlimited} tokens"
echo "Fixed SP Size: $FIXED_SP_SIZE"
echo "SP Backend  : $SP_BACKEND"
echo "CUDA Graph  : $CUDA_GRAPH_MODE"
echo "Run Label   : ${RUN_LABEL:-none}"
echo "Dynamic SP Size Strategy: $DYNAMIC_SP_SIZE_STRATEGY"
echo "Dynamic SP Bucket Preset: $DYNAMIC_SP_BUCKET_PRESET"
echo "Diagnostic Log Interval: $DIAGNOSTIC_LOG_INTERVAL"
echo "Slow Add Threshold: ${SLOW_ADD_THRESHOLD_MS}ms"
echo "Hierarchical Execution Trace: $HIERARCHICAL_EXECUTION_TRACE"
echo "Quantum Diagnostics: $QUANTUM_DIAGNOSTICS"
echo "Enforce Eager: $ENFORCE_EAGER"
echo "Model Path  : $MODEL_PATH"
echo "Rates       : ${RATES[*]}"
echo "================================================"

log_progress "=== NEW BATCH STARTED ==="
log_progress "Model: $MODEL_NAME | Dataset: $DATASET_NAME"
log_progress "Parallel: DP=$DP, SP=$SP, EP=$EP, TP=$TP | Scheduler: $SCHEDULER_ARCH | RouterPolicy: $ROUTER_POLICY | SPMasterSelector: $SP_MASTER_SELECTOR"
log_progress "SegSize=$SEG_SIZE | BatchSize=$BATCH_SIZE | MaxLen=$MAX_MODEL_LEN | MaxInput=${MAX_INPUT_LEN:-unlimited} | MaxRequestTokens=${MAX_REQUEST_TOKENS:-unlimited} | FixedSPSize=$FIXED_SP_SIZE"
log_progress "SPBackend=$SP_BACKEND"
log_progress "CUDAGraphMode=$CUDA_GRAPH_MODE"
log_progress "DynamicSPSizeStrategy=$DYNAMIC_SP_SIZE_STRATEGY"
log_progress "DynamicSPBucketPreset=$DYNAMIC_SP_BUCKET_PRESET"
log_progress "DiagnosticLogInterval=$DIAGNOSTIC_LOG_INTERVAL"
log_progress "SlowAddThresholdMs=$SLOW_ADD_THRESHOLD_MS"
log_progress "HierarchicalExecutionTrace=$HIERARCHICAL_EXECUTION_TRACE"
log_progress "QuantumDiagnostics=$QUANTUM_DIAGNOSTICS"
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
OVERALL_EXIT_CODE=0
for rate in "${RATES[@]}"; do
    TIMESTAMP=$(TZ='Asia/Shanghai' date "+%Y%m%d_%H%M%S")

    # 策略标识字符串 (缩写)
    # 路由缩写: LeastBatch->LB, LeastCache->LC, RoundRobin->RR
    rt_short=""
    case "$ROUTING_STRATEGY" in
        LeastBatch)      rt_short="LB" ;;
        LeastCache)      rt_short="LC" ;;
        RoundRobin)      rt_short="RR" ;;
        *)               rt_short="$ROUTING_STRATEGY" ;;
    esac
    # 调度缩写
    sc_short=""
    case "$SCHEDULER_ARCH" in
        legacy_global) sc_short="legacy" ;;
        hierarchical) sc_short="hier" ;;
        *)             sc_short="$SCHEDULER_ARCH" ;;
    esac
    case "$ROUTER_POLICY" in
        round_robin) rp_short="rpRR" ;;
        least_batch) rp_short="rpLB" ;;
        least_cache) rp_short="rpLC" ;;
        *)           rp_short="$ROUTER_POLICY" ;;
    esac
    # segment size 缩写 (65536->64k)
    seg_short=$(( SEG_SIZE / 1024 ))k
    # max-input-len 缩写
    maxin_tag=""
    if [[ -n "$MAX_INPUT_LEN" ]]; then
        maxin_tag="_maxin$(( MAX_INPUT_LEN / 1000 ))k"
    fi
    maxreq_tag=""
    if [[ "$MAX_REQUEST_TOKENS" -gt 0 ]]; then
        maxreq_tag="_maxreq${MAX_REQUEST_TOKENS}"
    fi
    # 可选标签
    extra_tags=""
    if [[ -n "$DISABLE_NON_UNIFORM_SPLIT" ]]; then
        extra_tags="${extra_tags}_uni"
    fi
    if [[ "$FIXED_SP_SIZE" -ne 0 ]]; then
        extra_tags="${extra_tags}_fsp${FIXED_SP_SIZE}"
    fi
    extra_tags="${extra_tags}_${SP_BACKEND}"
    if [[ "$CUDA_GRAPH_MODE" != "full" ]]; then
        extra_tags="${extra_tags}_${CUDA_GRAPH_MODE}"
    fi
    if [[ "$DYNAMIC_SP_SIZE_STRATEGY" != "legacy" ]]; then
        extra_tags="${extra_tags}_${DYNAMIC_SP_SIZE_STRATEGY}_${DYNAMIC_SP_BUCKET_PRESET}"
    fi
    if [[ "$QUANTUM_DIAGNOSTICS" -ne 0 ]]; then
        extra_tags="${extra_tags}_qdiag"
    fi
    if [[ -n "$RUN_LABEL" ]]; then
        extra_tags="${extra_tags}_${RUN_LABEL}"
    fi
    STRATEGY_STR="dp${DP}sp${SP}_seg${seg_short}_n${NUM_REQUESTS}_r${rate}_bs${BATCH_SIZE}_${rt_short}_${sc_short}_${rp_short}${maxin_tag}${maxreq_tag}${extra_tags}"

    CURRENT_LOG_DIR="$BASE_LOG_DIR/$MODEL_NAME/$DATASET_NAME/$STRATEGY_STR"
    mkdir -p "$CURRENT_LOG_DIR"

    LOG_FILE="$CURRENT_LOG_DIR/${TIMESTAMP}.log"
    JSON_FILE="$CURRENT_LOG_DIR/${TIMESTAMP}.jsonl"
    TRACE_FILE="$CURRENT_LOG_DIR/${TIMESTAMP}.hier_trace.jsonl"
    QUANTUM_FILE="$CURRENT_LOG_DIR/${TIMESTAMP}.quantum.jsonl"

    # 构建 Python 命令
    CMD=(
        python3 -u "$PYTHON_SCRIPT"
        --dataset csv
        --csv-path "$CSV_PATH"
        --max-request-tokens "$MAX_REQUEST_TOKENS"
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
        --request-metrics-log-path "$JSON_FILE"
        --segment-size "$SEG_SIZE"
        --sp-backend "$SP_BACKEND"
        --cuda-graph-mode "$CUDA_GRAPH_MODE"
        --scheduler-arch "$SCHEDULER_ARCH"
        --router-policy "$ROUTER_POLICY"
        --sp-master-selector "$SP_MASTER_SELECTOR"
        --fixed-sp-size "$FIXED_SP_SIZE"
        --dynamic-sp-size-strategy "$DYNAMIC_SP_SIZE_STRATEGY"
        --dynamic-sp-bucket-preset "$DYNAMIC_SP_BUCKET_PRESET"
        --diagnostic-log-interval "$DIAGNOSTIC_LOG_INTERVAL"
        --slow-add-threshold-ms "$SLOW_ADD_THRESHOLD_MS"
    )

    # 如果启用了 disable_non_uniform_split
    if [[ -n "$DISABLE_NON_UNIFORM_SPLIT" ]]; then
        CMD+=(--disable-non-uniform-split)
    fi

    if [[ "$ENFORCE_EAGER" -ne 0 ]]; then
        CMD+=(--enforce-eager)
    fi
    if [[ "$HIERARCHICAL_EXECUTION_TRACE" -ne 0 ]]; then
        CMD+=(
            --hierarchical-execution-trace
            --hierarchical-trace-log-path "$TRACE_FILE"
        )
    fi
    if [[ "$QUANTUM_DIAGNOSTICS" -ne 0 ]]; then
        CMD+=(
            --quantum-log-path "$QUANTUM_FILE"
        )
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
        echo "Max Request Tokens: ${MAX_REQUEST_TOKENS:-unlimited}"
        echo "Settings: $STRATEGY_STR"
        echo ""
        echo "================= Reproduce Command ================="
        echo "PYTHONUNBUFFERED=1 RAY_DEDUP_LOGS=0 ${CMD[*]}"
        echo "====================================================="
        echo ""
    } > "$LOG_FILE"

    echo ">>> Starting Rate: $rate | Output: $CURRENT_LOG_DIR"

    # 执行命令并捕获退出码
    cd "$ROOT_DIR"
    set -o pipefail
    PYTHONUNBUFFERED=1 RAY_DEDUP_LOGS=0 "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=$?
    set +o pipefail

    if [ $EXIT_CODE -eq 0 ]; then
        log_progress " <- Finished Rate: $rate. Status: SUCCESS"
    else
        log_progress " <- Finished Rate: $rate. Status: FAILED (Code $EXIT_CODE)"
        OVERALL_EXIT_CODE=$EXIT_CODE
    fi

    sleep 15
done

log_progress "=== BATCH FINISHED ==="
exit "$OVERALL_EXIT_CODE"
