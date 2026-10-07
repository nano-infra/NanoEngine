#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_ROOT="$SCRIPT_DIR"
cd "$SCRIPT_DIR"

# Machine-specific roots come from paths.env, the single place in this tree
# that may hold absolute paths. Existing environment variables win.
if [[ -f "$AE_ROOT/paths.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$AE_ROOT/paths.env"
    set +a
fi
require_env() {
    if [[ -z "${!1:-}" ]]; then
        echo "$1 is not set. Add it to $AE_ROOT/paths.env or export it." >&2
        exit 1
    fi
}

RUN_NAME="${RUN_NAME:-dpsk_issue01_dp4sp8_sp_policy}"
OUTPUT_ROOT="${OUTPUT_ROOT:-bench_logs/e2e}"
RUN_ROOT="$OUTPUT_ROOT/$RUN_NAME"

DPSK_MODEL_PATH="${DPSK_MODEL_PATH:-${MODEL_PATH:-${AE_DPSK_MODEL:-}}}"
KIMI_MODEL_PATH="${KIMI_MODEL_PATH:-${AE_KIMI_MODEL:-}}"
DATASET_PATH="${DATASET_PATH:-${AE_DATASET_MIXLONG_0326:-}/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv}"
require_env DPSK_MODEL_PATH
require_env KIMI_MODEL_PATH
require_env DATASET_PATH
RAY_ADDR="${RAY_ADDR:-10.102.252.174:6380}"
MASTER_ADDR="${MASTER_ADDR:-10.102.252.174:29500}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SEND_DURATION_SEC="${SEND_DURATION_SEC:-600}"
RATE_COOLDOWN_SEC="${RATE_COOLDOWN_SEC:-15}"
DRY_RUN="${DRY_RUN:-0}"
DPSK_REQUEST_RATES=(50 70 80 90)
KIMI_REQUEST_RATES=(40 60 80 100)
CASES=(bucket kimi_bucket)

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    printf 'ERROR: DRY_RUN must be 0 or 1; got %s\n' "$DRY_RUN" >&2
    exit 1
fi

export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_TC="${NCCL_IB_TC:-186}"
export SLIME_VISIBLE_DEVICES="${SLIME_VISIBLE_DEVICES:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export SLIME_GID_INDEX="${SLIME_GID_INDEX:-3}"
export SLIME_QP_NUM="${SLIME_QP_NUM:-4}"
export DEEPEP_SMS="${DEEPEP_SMS:-16}"
export DEEPEP_MAX_TOKENS_PER_RANK="${DEEPEP_MAX_TOKENS_PER_RANK:-256}"
export DEEPEP_ENABLE_MNNVL="${DEEPEP_ENABLE_MNNVL:-0}"
export DEEPEP_MODE="${DEEPEP_MODE:-auto}"
export NVSHMEM_QP_DEPTH="${NVSHMEM_QP_DEPTH:-1024}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export NANODEPLOY_LOG_DECODE_STEP_DETAIL=0

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

if [[ -e "$RUN_ROOT" && ! -d "$RUN_ROOT" ]]; then
    printf 'ERROR: output path is not a directory: %s\n' "$RUN_ROOT" >&2
    exit 1
fi

mkdir -p "$RUN_ROOT"
SUMMARY_FILE="$RUN_ROOT/run_summary.tsv"
if [[ -e "$SUMMARY_FILE" && ! -f "$SUMMARY_FILE" ]]; then
    printf 'ERROR: summary path is not a regular file: %s\n' "$SUMMARY_FILE" >&2
    exit 1
fi
if [[ ! -s "$SUMMARY_FILE" ]]; then
    printf 'policy\trate\tnum_requests\tstatus\tlog_dir\n' > "$SUMMARY_FILE"
fi

total_cases=$((${#DPSK_REQUEST_RATES[@]} + ${#KIMI_REQUEST_RATES[@]}))
case_index=0

for policy in "${CASES[@]}"; do
    if [[ "$policy" == "bucket" ]]; then
        model_name=dpsk
        model_path="$DPSK_MODEL_PATH"
        bucket_preset=deepseek_v3
        request_rates=("${DPSK_REQUEST_RATES[@]}")
    else
        model_name=kimi
        model_path="$KIMI_MODEL_PATH"
        bucket_preset=kimi_k2
        request_rates=("${KIMI_REQUEST_RATES[@]}")
    fi

    for rate in "${request_rates[@]}"; do
        case_index=$((case_index + 1))
        case_dir="$RUN_ROOT/$policy/rate_$rate"

        num_requests=$(awk -v rate="$rate" -v duration="$SEND_DURATION_SEC" \
            'BEGIN {printf "%d", int(rate * duration + 0.5)}')

        summary_status="$(
            awk -F '\t' \
                -v wanted_policy="$policy" \
                -v wanted_rate="$rate" \
                -v wanted_requests="$num_requests" '
                NR > 1 && $1 == wanted_policy && $2 == wanted_rate {
                    status = $4
                    requests = $3
                }
                END {
                    if (requests == wanted_requests && status != "") {
                        print status
                    }
                }
            ' "$SUMMARY_FILE"
        )"

        command_matches_preset=0
        if [[ -f "$case_dir/command.txt" ]] \
            && grep -Fq -- \
                "--dynamic-sp-bucket-preset $bucket_preset" \
                "$case_dir/command.txt"; then
            command_matches_preset=1
        fi

        if [[ "$summary_status" == "ok" && "$command_matches_preset" -eq 1 ]]; then
            printf '\n[%d/%d] model=%s preset=%s rate=%s num_requests=%s already completed; skipping\n' \
                "$case_index" "$total_cases" "$model_name" "$bucket_preset" \
                "$rate" "$num_requests"
            continue
        fi
        if [[ "$summary_status" == "ok" ]]; then
            printf '\n[%d/%d] model=%s preset=%s has stale completed output; rerunning\n' \
                "$case_index" "$total_cases" "$model_name" "$bucket_preset"
        fi

        # Recover the first point produced by the pre-resume script, which
        # crashed before it could append an OK row to run_summary.tsv.
        if [[ -z "$summary_status" \
            && "$command_matches_preset" -eq 1 \
            && -s "$case_dir/driver.log" \
            && -s "$case_dir/itl_samples.jsonl" ]] \
            && awk -v expected_requests="$num_requests" '
                $0 ~ ("Requests sent: " expected_requests "$") { sent = 1 }
                $0 ~ ("Requests completed: " expected_requests "$") { completed = 1 }
                $0 ~ ("Saved " expected_requests " ITL samples to ") { saved = 1 }
                END { exit !(sent && completed && saved) }
            ' "$case_dir/driver.log"; then
            printf '%s\t%s\t%s\tok\t%s\n' \
                "$policy" "$rate" "$num_requests" "$case_dir" \
                >> "$SUMMARY_FILE"
            printf '\n[%d/%d] model=%s preset=%s rate=%s recovered from completed log; skipping\n' \
                "$case_index" "$total_cases" "$model_name" "$bucket_preset" "$rate"
            continue
        fi

        mkdir -p "$case_dir"

        policy_args=(
            --dynamic-sp-size-strategy bucket
            --dynamic-sp-bucket-preset "$bucket_preset"
        )

        command=(
            "$PYTHON_BIN" -u start-e2e/nano/bench_serving_overhead.py
            --dataset csv
            --csv-path "$DATASET_PATH"
            --max-request-tokens 0
            --num-requests "$num_requests"
            --request-rate "$rate"
            --sp 8
            --dp 4
            --ep 32
            --tp 1
            --max-num-seqs 256
            --gpu-memory-limit-gb 141
            --gpu-memory-utilization 0.9
            --max-model-len 1000000
            --max-input-len 1000000
            --dummy-prefill
            --ray-address "$RAY_ADDR"
            --master-address "$MASTER_ADDR"
            --loop-count 16
            --model-path "$model_path"
            --routing-strategy LeastBatch
            --itl-log-path "$case_dir/itl_samples.jsonl"
            --segment-size 65536
            --sp-backend hao_basic
            --cuda-graph-mode full
            --scheduler-arch legacy_global
            --fixed-sp-size 0
            --enable-dynamic-sp-size
            "${policy_args[@]}"
        )

        {
            printf '%q ' "${command[@]}"
            printf '\n'
        } > "$case_dir/command.txt"

        printf '\n[%d/%d] model=%s preset=%s rate=%s num_requests=%s\n' \
            "$case_index" "$total_cases" "$model_name" "$bucket_preset" \
            "$rate" "$num_requests"

        if [[ "$DRY_RUN" == "1" ]]; then
            printf 'DRY_RUN: '
            printf '%q ' "${command[@]}"
            printf '\n'
            printf '%s\t%s\t%s\tplanned\t%s\n' \
                "$policy" "$rate" "$num_requests" "$case_dir" \
                >> "$SUMMARY_FILE"
            continue
        fi

        set +e
        "${command[@]}" 2>&1 | tee "$case_dir/driver.log"
        run_status=${PIPESTATUS[0]}
        set -e

        if [[ "$run_status" -eq 0 ]]; then
            status=ok
        else
            status=failed
        fi
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$policy" "$rate" "$num_requests" "$status" "$case_dir" \
            >> "$SUMMARY_FILE"

        if [[ "$run_status" -ne 0 ]]; then
            printf 'ERROR: %s bucket rate %s failed; see %s/driver.log\n' \
                "$model_name" "$rate" "$case_dir" >&2
            exit "$run_status"
        fi

        if [[ "$case_index" -lt "$total_cases" ]]; then
            sleep "$RATE_COOLDOWN_SEC"
        fi
    done
done

if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDRY_RUN complete; no benchmark was launched. Summary: %s\n' \
        "$SUMMARY_FILE"
else
    printf '\nCompleted all cases. Summary: %s\n' "$SUMMARY_FILE"
fi
