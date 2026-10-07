#!/usr/bin/env bash
# Run this script concurrently on every node participating in the DeepEP job.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MICROBENCH_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-${MICROBENCH_DIR}/results/deepep}"
BENCHMARK_PY="${SCRIPT_DIR}/run_one_low_latency.py"
VENDORED_TEST_PY="${SCRIPT_DIR}/third_party/deepep/test_low_latency.py"
VENDORED_UTILS_PY="${SCRIPT_DIR}/third_party/deepep/utils.py"
EXPECTED_DEEPEP_VERSION="1.2.1+73b6ea4"

# IMPORTANT: DeepEP tests/utils.py uses WORLD_SIZE for the number of nodes and
# RANK for the node rank.  These are not the total GPU rank count/global rank.
NODE_RANK="${RANK:-${NODE_RANK:-0}}"
NUM_NODES="${WORLD_SIZE:-}"
LOCAL_PROCESSES="${EP_TEST_NUM_PROCESSES:-8}"
BASE_MASTER_PORT="${MASTER_PORT:-8361}"
# Match DeepSeek-V3's n_routed_experts. DeepEP's stock test default (288) is a
# generic kernel-test setting rather than the model configuration.
NUM_EXPERTS="${EP_TEST_NUM_EXPERTS:-256}"
TOKEN_TIMEOUT_SECONDS="${EP_TOKEN_TIMEOUT_SECONDS:-300}"
TOKEN_MAX_ATTEMPTS="${EP_TOKEN_MAX_ATTEMPTS:-3}"
RETRY_DELAY_SECONDS="${EP_TOKEN_RETRY_DELAY_SECONDS:-10}"
RESUME="${EP_RESUME:-0}"
CURRENT_BENCHMARK_PGID=""

cleanup_current_benchmark() {
    if [[ -z "${CURRENT_BENCHMARK_PGID}" ]]; then
        return
    fi

    if kill -0 -- "-${CURRENT_BENCHMARK_PGID}" 2>/dev/null; then
        echo "Stopping DeepEP worker group ${CURRENT_BENCHMARK_PGID} ..." >&2
        kill -TERM -- "-${CURRENT_BENCHMARK_PGID}" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            if ! kill -0 -- "-${CURRENT_BENCHMARK_PGID}" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 -- "-${CURRENT_BENCHMARK_PGID}" 2>/dev/null; then
            kill -KILL -- "-${CURRENT_BENCHMARK_PGID}" 2>/dev/null || true
        fi
    fi
    wait "${CURRENT_BENCHMARK_PGID}" 2>/dev/null || true
    CURRENT_BENCHMARK_PGID=""
}

trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit_code=$?; trap - EXIT; cleanup_current_benchmark; exit "${exit_code}"' EXIT

if [[ -z "${TOKENS:-}" ]]; then
    echo "TOKENS must be set by the figure preset or caller" >&2
    exit 2
fi
NORMALIZED_TOKENS="${TOKENS//$'\n'/ }"
read -r -a TOKEN_ARRAY <<< "${NORMALIZED_TOKENS}"
if [[ -n "${EP_EXPECTED_TOKEN_COUNT:-}" ]]; then
    if [[ ! "${EP_EXPECTED_TOKEN_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
        echo "EP_EXPECTED_TOKEN_COUNT must be a positive integer" >&2
        exit 2
    fi
    if (( ${#TOKEN_ARRAY[@]} != EP_EXPECTED_TOKEN_COUNT )); then
        echo "Expected ${EP_EXPECTED_TOKEN_COUNT} token values, parsed ${#TOKEN_ARRAY[@]}: ${TOKEN_ARRAY[*]}" >&2
        exit 2
    fi
fi

for required_path in "${BENCHMARK_PY}" "${VENDORED_TEST_PY}" "${VENDORED_UTILS_PY}"; do
    if [[ ! -f "${required_path}" ]]; then
        echo "Required DeepEP benchmark artifact not found: ${required_path}" >&2
        exit 2
    fi
done
"${PYTHON_BIN}" - "${EXPECTED_DEEPEP_VERSION}" <<'PY'
import importlib.metadata
import sys

import deep_ep  # noqa: F401

expected = sys.argv[1]
actual = importlib.metadata.version("deep_ep")
if actual != expected:
    raise SystemExit(
        f"DeepEP version mismatch: expected {expected}, found {actual}"
    )
print(f"DeepEP package: {actual}")
PY
if [[ ! "${NODE_RANK}" =~ ^[0-9]+$ ]] || [[ ! "${NUM_NODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RANK must be a non-negative node rank and WORLD_SIZE must be set to a positive node count" >&2
    exit 2
fi
if [[ ! "${LOCAL_PROCESSES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EP_TEST_NUM_PROCESSES must be positive" >&2
    exit 2
fi
if (( NODE_RANK >= NUM_NODES )); then
    echo "RANK=${NODE_RANK} must be smaller than WORLD_SIZE=${NUM_NODES}" >&2
    exit 2
fi
if [[ ! "${NUM_EXPERTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EP_TEST_NUM_EXPERTS must be positive" >&2
    exit 2
fi
if [[ ! "${TOKEN_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EP_TOKEN_TIMEOUT_SECONDS must be positive" >&2
    exit 2
fi
if [[ ! "${TOKEN_MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EP_TOKEN_MAX_ATTEMPTS must be positive" >&2
    exit 2
fi
if [[ ! "${RETRY_DELAY_SECONDS}" =~ ^[0-9]+$ ]]; then
    echo "EP_TOKEN_RETRY_DELAY_SECONDS must be non-negative" >&2
    exit 2
fi
if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
    echo "EP_RESUME must be 0 or 1" >&2
    exit 2
fi
total_ranks=$((NUM_NODES * LOCAL_PROCESSES))
if (( NUM_EXPERTS % total_ranks != 0 )); then
    echo "EP_TEST_NUM_EXPERTS=${NUM_EXPERTS} must be divisible by total ranks=${total_ranks}" >&2
    exit 2
fi
if [[ ! "${BASE_MASTER_PORT}" =~ ^[1-9][0-9]*$ ]] || (( BASE_MASTER_PORT > 65535 )); then
    echo "MASTER_PORT must be an integer in [1, 65535]" >&2
    exit 2
fi
if (( NUM_NODES > 1 )) && [[ -z "${MASTER_ADDR:-}" ]]; then
    echo "MASTER_ADDR must be set for a multi-node run" >&2
    exit 2
fi
if [[ ${#TOKEN_ARRAY[@]} -eq 0 ]]; then
    echo "TOKENS did not contain any token counts" >&2
    exit 2
fi
if ! command -v setsid >/dev/null 2>&1; then
    echo "setsid is required to manage and clean up DeepEP worker processes" >&2
    exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
    echo "timeout is required to bound each DeepEP token attempt" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}/logs"
export RANK="${NODE_RANK}"
export WORLD_SIZE="${NUM_NODES}"
export EP_TEST_NUM_PROCESSES="${LOCAL_PROCESSES}"
export EP_TEST_NUM_EXPERTS="${NUM_EXPERTS}"

echo "DeepEP benchmark: ${VENDORED_TEST_PY}"
echo "Node: ${NODE_RANK}/${NUM_NODES}; local processes: ${LOCAL_PROCESSES}"
echo "Total ranks: ${total_ranks}; experts: ${NUM_EXPERTS}; experts/rank: $((NUM_EXPERTS / total_ranks))"
echo "Master: ${MASTER_ADDR:-127.0.0.1}; base port: ${BASE_MASTER_PORT}"
echo "Token count: ${#TOKEN_ARRAY[@]}"
echo "Tokens: ${TOKEN_ARRAY[*]}"
echo "Per-token timeout: ${TOKEN_TIMEOUT_SECONDS}s; max attempts: ${TOKEN_MAX_ATTEMPTS}; resume: ${RESUME}"
echo "Output: ${OUTPUT_DIR}"

log_has_complete_metrics() {
    local log_path="$1"
    local global_rank="$2"
    [[ -f "${log_path}" ]] || return 1
    grep -q '^DEEPEP_SWEEP_EXIT_CODE=0$' "${log_path}" || return 1
    grep -Fq "[rank ${global_rank}] Dispatch + combine bandwidth:" "${log_path}" || return 1
    grep -Fq "[rank ${global_rank}] Dispatch bandwidth:" "${log_path}" || return 1
    grep -Fq "[rank ${global_rank}] Dispatch send/recv time:" "${log_path}" || return 1
}

token_is_complete_on_all_nodes() {
    local token="$1"
    local node
    local global_rank
    local node_log
    for ((node = 0; node < NUM_NODES; node++)); do
        global_rank=$((node * LOCAL_PROCESSES))
        node_log="${OUTPUT_DIR}/logs/node${node}_tokens_${token}.log"
        log_has_complete_metrics "${node_log}" "${global_rank}" || return 1
    done
}

for token in "${TOKEN_ARRAY[@]}"; do
    if [[ ! "${token}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid positive token count: ${token}" >&2
        exit 2
    fi
    if (( BASE_MASTER_PORT + token > 65535 )); then
        echo "MASTER_PORT + token exceeds 65535: ${BASE_MASTER_PORT} + ${token}" >&2
        exit 2
    fi

    log_file="${OUTPUT_DIR}/logs/node${NODE_RANK}_tokens_${token}.log"
    if [[ "${RESUME}" == "1" ]] && token_is_complete_on_all_nodes "${token}"; then
        echo "[node ${NODE_RANK}] token=${token} already complete on all nodes; skipping"
        continue
    fi

    export EP_TEST_NUM_TOKENS="${token}"
    # A distinct rendezvous port avoids reuse races between consecutive runs.
    export MASTER_PORT="$((BASE_MASTER_PORT + token))"

    benchmark_rc=1
    for ((attempt = 1; attempt <= TOKEN_MAX_ATTEMPTS; attempt++)); do
        echo "[node ${NODE_RANK}] benchmarking token=${token}, port=${MASTER_PORT}, attempt=${attempt}/${TOKEN_MAX_ATTEMPTS}"
        setsid timeout --signal=TERM --kill-after=30s "${TOKEN_TIMEOUT_SECONDS}s" \
            env PYTHONUNBUFFERED=1 \
            "${PYTHON_BIN}" "${BENCHMARK_PY}" \
            > >(tee "${log_file}") 2>&1 &
        CURRENT_BENCHMARK_PGID=$!
        set +e
        wait "${CURRENT_BENCHMARK_PGID}"
        benchmark_rc=$?
        set -e
        cleanup_current_benchmark
        printf '\nDEEPEP_SWEEP_EXIT_CODE=%d\n' "${benchmark_rc}" | tee -a "${log_file}"
        if [[ ${benchmark_rc} -eq 0 ]]; then
            break
        fi
        if [[ ${benchmark_rc} -eq 124 ]]; then
            echo "[node ${NODE_RANK}] token=${token} attempt=${attempt} timed out after ${TOKEN_TIMEOUT_SECONDS}s" >&2
        else
            echo "[node ${NODE_RANK}] token=${token} attempt=${attempt} failed with exit code ${benchmark_rc}" >&2
        fi
        if (( attempt < TOKEN_MAX_ATTEMPTS )); then
            echo "[node ${NODE_RANK}] retrying token=${token} in ${RETRY_DELAY_SECONDS}s" >&2
            sleep "${RETRY_DELAY_SECONDS}"
        fi
    done
    if [[ ${benchmark_rc} -ne 0 ]]; then
        echo "[node ${NODE_RANK}] token=${token} failed after ${TOKEN_MAX_ATTEMPTS} attempts" >&2
        exit "${benchmark_rc}"
    fi
done

first_global_rank=$((NODE_RANK * LOCAL_PROCESSES))
summary_file="${OUTPUT_DIR}/node${NODE_RANK}_summary_rank${first_global_rank}.csv"
"${PYTHON_BIN}" "${SCRIPT_DIR}/parse_low_latency_logs.py" \
    --log-dir "${OUTPUT_DIR}/logs" \
    --node-rank "${NODE_RANK}" \
    --local-processes "${LOCAL_PROCESSES}" \
    --output "${summary_file}" \
    --tokens "${TOKEN_ARRAY[@]}"
parser_rc=$?

if [[ ${parser_rc} -ne 0 ]]; then
    exit "${parser_rc}"
fi
echo "Completed successfully: ${summary_file}"
