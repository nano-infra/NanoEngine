#!/usr/bin/env bash

# Per-node Figure 5 DeepEP worker. launch_deepep.py starts this script on every
# node; direct invocation remains available for debugging.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FIG5_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
AE_ROOT="$(cd -- "${FIG5_DIR}/.." && pwd)"
SWEEP_SH="${AE_ROOT}/microbench/deepep/run_low_latency_sweep.sh"

SNAPSHOT=""
NUM_NODES=""
NODE_RANK=""
MASTER_ADDRESS=""
MASTER_BASE_PORT=18361
RUN_ID="ae_run1"
PYTHON_COMMAND="python3"
RESULT_DIR=""
TOKEN_TIMEOUT_SECONDS=300
TOKEN_MAX_ATTEMPTS=3
MAX_CASES=0

usage() {
    cat <<'EOF'
Usage: ./run_deepep.sh --snapshot FILE --num-nodes N --node-rank R --master-addr IP [options]

This is the low-level per-node worker. For normal AE runs, invoke
launch_deepep.py once on node 0 so that it starts every rank over SSH. Direct
invocation is available for debugging; all nodes must use the same snapshot,
node count, master address, port, and run ID. Change only node rank.

Required:
  --snapshot FILE      DeepEP rank_snapshot_time60.json
  --num-nodes N        Number of nodes (2 or 4)
  --node-rank R        This node's rank in [0, N-1]
  --master-addr IP     Address of node rank 0

Options:
  --master-port PORT   Base rendezvous port (default: 18361)
  --run-id ID          Shared run name (default: ae_run1)
  --python PATH        Python command (default: python3)
  --output-dir DIR     Override the result directory
  --max-cases N        Measure at most N evenly spaced snapshot points
                       (default: 0, meaning all points)
  --token-timeout S    Timeout for one token attempt (default: 300 seconds)
  --max-attempts N     Attempts per token after timeout/failure (default: 3)
  -h, --help           Show this help
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
        --snapshot)
            require_option_value "$@"
            SNAPSHOT="$2"
            shift 2
            ;;
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
        --output-dir)
            require_option_value "$@"
            RESULT_DIR="$2"
            shift 2
            ;;
        --max-cases)
            require_option_value "$@"
            MAX_CASES="$2"
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

if [[ -z "$SNAPSHOT" || ! -f "$SNAPSHOT" ]]; then
    echo "Error: --snapshot must name an existing file." >&2
    exit 2
fi
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
if [[ ! "$MAX_CASES" =~ ^[0-9]+$ ]] || (( MAX_CASES == 1 )); then
    echo "Error: --max-cases must be 0 or an integer of at least 2." >&2
    exit 2
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Error: --run-id contains unsupported characters." >&2
    exit 2
fi
if [[ ! -x "$SWEEP_SH" ]]; then
    echo "Error: shared DeepEP engine not found: $SWEEP_SH" >&2
    exit 2
fi
command -v "$PYTHON_COMMAND" >/dev/null 2>&1 || {
    echo "Error: Python not found: $PYTHON_COMMAND" >&2
    exit 2
}
LOCAL_PROCESSES="${EP_TEST_NUM_PROCESSES:-8}"
if [[ ! "$LOCAL_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: EP_TEST_NUM_PROCESSES must be a positive integer." >&2
    exit 2
fi

TOKENS="$($PYTHON_COMMAND - "$SNAPSHOT" "$MAX_CASES" "$NUM_NODES" \
    "$LOCAL_PROCESSES" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as input_file:
    metadata = json.load(input_file)
max_cases = int(sys.argv[2])
expected_rank_count = int(sys.argv[3]) * int(sys.argv[4])
if int(metadata["rank_count"]) != expected_rank_count:
    raise SystemExit(
        "snapshot rank_count does not match num_nodes * local_processes: "
        f'{metadata["rank_count"]} != {expected_rank_count}'
    )
if int(metadata["time_percent"]) != 60:
    raise SystemExit("Figure 5 requires the time=60% snapshot")
values = [
    int(value)
    for value in metadata["deepep"]["microbenchmark_batch_cases"]
]
if not values or len(values) != len(set(values)) or any(value <= 0 for value in values):
    raise SystemExit("snapshot contains invalid DeepEP microbenchmark cases")
values = sorted(values)
if max_cases and max_cases < len(values):
    indices = [
        index * (len(values) - 1) // (max_cases - 1)
        for index in range(max_cases)
    ]
    values = [values[index] for index in indices]
print(" ".join(map(str, values)))
PY
)"
read -r -a TOKEN_VALUES <<< "$TOKENS"

if [[ -z "$RESULT_DIR" ]]; then
    RESULT_DIR="${FIG5_DIR}/results/deepep/${RUN_ID}_${NUM_NODES}nodes"
fi
mkdir -p "$RESULT_DIR"

if (( NODE_RANK == 0 )); then
    "$PYTHON_COMMAND" - "$SNAPSHOT" "$TOKENS" "$NUM_NODES" \
        "$LOCAL_PROCESSES" "$RESULT_DIR/fig5_deepep_config.json" <<'PY'
import json
import os
import sys
from pathlib import Path

snapshot_path = Path(sys.argv[1]).expanduser().resolve()
with snapshot_path.open(encoding="utf-8") as input_file:
    metadata = json.load(input_file)
snapshot_cases = sorted(
    int(value)
    for value in metadata["deepep"]["microbenchmark_batch_cases"]
)
measured_cases = [int(value) for value in sys.argv[2].split()]
config = {
    "snapshot": str(snapshot_path),
    "snapshot_cases": snapshot_cases,
    "measured_cases": measured_cases,
    "num_nodes": int(sys.argv[3]),
    "local_processes": int(sys.argv[4]),
    "interpolation_required": measured_cases != snapshot_cases,
}
output = Path(sys.argv[5]).expanduser().resolve()
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
temporary.replace(output)
print(f"Figure 5 DeepEP config: {output}")
PY
fi

if (( NODE_RANK == 0 )) && [[ "$MASTER_ADDRESS" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    if command -v ip >/dev/null 2>&1 && ! ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1 | grep -Fxq "$MASTER_ADDRESS"; then
        echo "Error: --master-addr ${MASTER_ADDRESS} is not an IPv4 address on node rank 0." >&2
        exit 2
    fi
fi
for token in "${TOKEN_VALUES[@]}"; do
    if (( MASTER_BASE_PORT + token > 65535 )); then
        echo "Error: --master-port + token exceeds 65535: ${MASTER_BASE_PORT} + ${token}." >&2
        exit 2
    fi
done
if (( NODE_RANK == 0 )) && command -v ss >/dev/null 2>&1; then
    for token in "${TOKEN_VALUES[@]}"; do
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
export OUTPUT_DIR="$RESULT_DIR"
export PYTHON_BIN="$PYTHON_COMMAND"
export EP_EXPECTED_TOKEN_COUNT="${#TOKEN_VALUES[@]}"
export EP_TOKEN_TIMEOUT_SECONDS="$TOKEN_TIMEOUT_SECONDS"
export EP_TOKEN_MAX_ATTEMPTS="$TOKEN_MAX_ATTEMPTS"
export EP_RESUME=1

export EP_TEST_NUM_PROCESSES="$LOCAL_PROCESSES"
export EP_TEST_NUM_EXPERTS="${EP_TEST_NUM_EXPERTS:-256}"
export EP_TEST_HIDDEN="${EP_TEST_HIDDEN:-7168}"
export EP_TEST_NUM_TOPK="${EP_TEST_NUM_TOPK:-8}"
export EP_TEST_SEED="${EP_TEST_SEED:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Keep caller-provided cluster settings. These defaults match the Nano H200
# environment used for the paper and may be overridden before invoking us.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_TC="${NCCL_IB_TC:-186}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-120}"
export NVSHMEM_HCA_LIST="${NVSHMEM_HCA_LIST:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export NVSHMEM_IB_GID_INDEX="${NVSHMEM_IB_GID_INDEX:-3}"
export NVSHMEM_IBGDA_NUM_RC_PER_PE="${NVSHMEM_IBGDA_NUM_RC_PER_PE:-8}"
export NVSHMEM_IB_TRAFFIC_CLASS="${NVSHMEM_IB_TRAFFIC_CLASS:-186}"
export NVSHMEM_DISABLE_NVLS="${NVSHMEM_DISABLE_NVLS:-1}"

exec bash "$SWEEP_SH"
