#!/usr/bin/env bash

# Per-node Figure 3 DeepEP worker. launch_deepep.py starts this script on every
# node; direct invocation remains available for debugging.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FIG3_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
AE_ROOT="$(cd -- "${FIG3_DIR}/.." && pwd)"
SWEEP_SH="${AE_ROOT}/microbench/deepep/run_low_latency_sweep.sh"

NUM_NODES=""
NODE_RANK=""
MASTER_ADDRESS=""
MASTER_BASE_PORT=18361
RUN_ID="ae_run1"
PYTHON_COMMAND="python3"
RESULT_DIR=""
TOKEN_TIMEOUT_SECONDS=300
TOKEN_MAX_ATTEMPTS=3
BATCH_SIZE_LIST=""

usage() {
    cat <<'EOF'
Usage: ./run_deepep.sh --num-nodes N --node-rank R --master-addr IP [options]

This is the low-level per-node worker. For normal AE runs, invoke
launch_deepep.py once on node 0 so that it starts every rank over SSH. Direct
invocation is available for debugging; all nodes must use the same batch-size
list and only the node rank may change.

Required:
  --num-nodes N       Number of nodes: 2 for comparison, 4 for paper setup
  --node-rank R       This node's rank in [0, N-1]
  --master-addr IP    bond0 IPv4 address of node rank 0

Options:
  --master-port PORT  Base rendezvous port (default: 18361)
  --run-id ID         Shared run name (default: ae_run1)
  --python PATH       Python command (default: python3)
  --batch-sizes LIST  Quoted space- or comma-separated batch sizes per GPU
                      (default: full paper sweep)
  --output-dir DIR    Override the result directory
  --token-timeout S   Timeout for one token attempt (default: 300 seconds)
  --max-attempts N    Attempts per token after timeout/failure (default: 3)
  -h, --help          Show this help
EOF
}

require_option_value() {
    if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Error: $1 requires a value." >&2
        usage >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-nodes)
            require_option_value "$@"
            NUM_NODES="$2"
            shift 2
            ;;
        --node-rank)
            require_option_value "$@"
            NODE_RANK="$2"
            shift 2
            ;;
        --master-addr)
            require_option_value "$@"
            MASTER_ADDRESS="$2"
            shift 2
            ;;
        --master-port)
            require_option_value "$@"
            MASTER_BASE_PORT="$2"
            shift 2
            ;;
        --run-id)
            require_option_value "$@"
            RUN_ID="$2"
            shift 2
            ;;
        --python)
            require_option_value "$@"
            PYTHON_COMMAND="$2"
            shift 2
            ;;
        --batch-sizes)
            require_option_value "$@"
            BATCH_SIZE_LIST="$2"
            shift 2
            ;;
        --output-dir)
            require_option_value "$@"
            RESULT_DIR="$2"
            shift 2
            ;;
        --token-timeout)
            require_option_value "$@"
            TOKEN_TIMEOUT_SECONDS="$2"
            shift 2
            ;;
        --max-attempts)
            require_option_value "$@"
            TOKEN_MAX_ATTEMPTS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "$NUM_NODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --num-nodes must be a positive integer." >&2
    exit 2
fi
if [[ ! "$NODE_RANK" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NUM_NODES )); then
    echo "Error: --node-rank must be an integer in [0, $((NUM_NODES - 1))]." >&2
    exit 2
fi
if [[ -z "$MASTER_ADDRESS" ]]; then
    echo "Error: --master-addr is required." >&2
    exit 2
fi
if [[ ! "$MASTER_BASE_PORT" =~ ^[1-9][0-9]*$ ]] || (( MASTER_BASE_PORT > 65535 )); then
    echo "Error: --master-port must be an integer in [1, 65535]." >&2
    exit 2
fi
if [[ ! "$TOKEN_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --token-timeout must be a positive integer." >&2
    exit 2
fi
if [[ ! "$TOKEN_MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --max-attempts must be a positive integer." >&2
    exit 2
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Error: --run-id contains unsupported characters." >&2
    exit 2
fi
if [[ ! -x "$SWEEP_SH" ]]; then
    echo "Error: DeepEP sweep launcher not found or not executable: $SWEEP_SH" >&2
    exit 2
fi

if [[ -z "$BATCH_SIZE_LIST" ]]; then
    BATCH_SIZE_VALUES=({1..8} {16..256..8})
else
    BATCH_SIZE_LIST="${BATCH_SIZE_LIST//,/ }"
    read -r -a BATCH_SIZE_VALUES <<< "$BATCH_SIZE_LIST"
fi
if (( ${#BATCH_SIZE_VALUES[@]} == 0 )); then
    echo "Error: --batch-sizes must contain at least one value." >&2
    exit 2
fi
for batch_size in "${BATCH_SIZE_VALUES[@]}"; do
    if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: every --batch-sizes value must be a positive integer: $batch_size" >&2
        exit 2
    fi
done
TOKENS="${BATCH_SIZE_VALUES[*]}"
EXPECTED_TOKEN_COUNT="${#BATCH_SIZE_VALUES[@]}"
DEFAULT_NCCL_DEBUG="WARN"

if [[ -z "$RESULT_DIR" ]]; then
    RESULT_DIR="${FIG3_DIR}/results/deepep/${RUN_ID}_${NUM_NODES}nodes"
fi

# Fail before spawning workers if rank 0 was given another node's address.
if (( NODE_RANK == 0 )) && [[ "$MASTER_ADDRESS" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    if command -v ip >/dev/null 2>&1 && ! ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1 | grep -Fxq "$MASTER_ADDRESS"; then
        echo "Error: --master-addr ${MASTER_ADDRESS} is not an IPv4 address on node rank 0." >&2
        echo "If this is a worker node, keep --master-addr unchanged and set --node-rank to this node's nonzero rank." >&2
        echo "Otherwise query the rank-0 address with: ip -4 -o addr show dev bond0" >&2
        exit 2
    fi
fi

# The sweep uses the base port plus each token count.
if (( NODE_RANK == 0 )) && command -v ss >/dev/null 2>&1; then
    for token in $TOKENS; do
        token_port=$((MASTER_BASE_PORT + token))
        if ss -H -ltn "sport = :${token_port}" | grep -q .; then
            echo "Error: rendezvous port ${token_port} is already in use on node rank 0." >&2
            echo "Stop the old run or choose another --master-port." >&2
            exit 2
        fi
    done
fi

export MASTER_ADDR="$MASTER_ADDRESS"
export MASTER_PORT="$MASTER_BASE_PORT"
export WORLD_SIZE="$NUM_NODES"
export RANK="$NODE_RANK"
export TOKENS
export EP_EXPECTED_TOKEN_COUNT="$EXPECTED_TOKEN_COUNT"
export EP_TOKEN_TIMEOUT_SECONDS="$TOKEN_TIMEOUT_SECONDS"
export EP_TOKEN_MAX_ATTEMPTS="$TOKEN_MAX_ATTEMPTS"
export EP_RESUME=1
export OUTPUT_DIR="$RESULT_DIR"
export PYTHON_BIN="$PYTHON_COMMAND"
export EP_TEST_NUM_PROCESSES=8
export EP_TEST_NUM_EXPERTS=256
export EP_TEST_HIDDEN=7168
export EP_TEST_NUM_TOPK=8
export EP_TEST_SEED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_TC="${NCCL_IB_TC:-186}"
export NCCL_DEBUG="${NCCL_DEBUG:-${DEFAULT_NCCL_DEBUG}}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-120}"
export NVSHMEM_HCA_LIST="${NVSHMEM_HCA_LIST:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export NVSHMEM_IB_GID_INDEX="${NVSHMEM_IB_GID_INDEX:-3}"
export NVSHMEM_IBGDA_NUM_RC_PER_PE="${NVSHMEM_IBGDA_NUM_RC_PER_PE:-8}"
export NVSHMEM_IB_TRAFFIC_CLASS="${NVSHMEM_IB_TRAFFIC_CLASS:-186}"
export NVSHMEM_DISABLE_NVLS="${NVSHMEM_DISABLE_NVLS:-1}"

exec bash "$SWEEP_SH"
