#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
case "$SCRIPT_PATH" in
  /*) ;;
  *) SCRIPT_PATH="$PWD/$SCRIPT_PATH" ;;
esac
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/asplos-camera-august/nano-profile-32x8k-16gpu}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/Kimi-K2-Instruct-0905}"
MASTER_ADDRESS="${MASTER_ADDRESS:-10.102.252.174:29500}"
RAY_ADDRESS="${RAY_ADDRESS:-10.102.252.174:7789}"

NUM_SEQS="${NUM_SEQS:-512}"
SEQ_LEN="${SEQ_LEN:-8192}"
MAX_TOKENS="${MAX_TOKENS:-64}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_SEND_SEQS="${MAX_NUM_SEND_SEQS:-32}"
MAX_NUM_RECV_SEQS="${MAX_NUM_RECV_SEQS:-34}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
LOOP_COUNT="${LOOP_COUNT:-16}"
PROFILER_START_STEP="${PROFILER_START_STEP:-3}"
PROFILING_STEP="${PROFILING_STEP:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MOE_ROUTING_SIMULATION_STRATEGY="${MOE_ROUTING_SIMULATION_STRATEGY:-uniform_random}"
SEED="${SEED:-0}"

SP_BACKEND="${SP_BACKEND:-hao_basic}"
SEGMENT_SIZE="${SEGMENT_SIZE:-65536}"
LONG_REQUEST_SP_THRESHOLD="${LONG_REQUEST_SP_THRESHOLD:-100000}"
DRY_RUN="${DRY_RUN:-0}"

export PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH:-}"
export LD_LIBRARY_PATH="$REPO_ROOT/build/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/build/lib:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export SLIME_VISIBLE_DEVICES="${SLIME_VISIBLE_DEVICES:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export SLIME_GID_INDEX="${SLIME_GID_INDEX:-3}"
export SLIME_QP_NUM="${SLIME_QP_NUM:-4}"

# Ray Client and GCS traffic must bypass HTTP proxies.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

if [[ ! -d "$MODEL_PATH" ]]; then
  printf 'MODEL_PATH not found: %s\n' "$MODEL_PATH" >&2
  exit 1
fi
if [[ "$NUM_SEQS" -ne 512 || "$SEQ_LEN" -ne 8192 ]]; then
  printf 'This profile requires NUM_SEQS=512 and SEQ_LEN=8192; got %s and %s\n' \
    "$NUM_SEQS" "$SEQ_LEN" >&2
  exit 1
fi

cases=(dp2sp8_ep16 dp16_ep16)
if (( $# > 0 )); then
  cases=("$@")
fi

mkdir -p "$OUTPUT_ROOT"
SUMMARY_PATH="$OUTPUT_ROOT/summary.tsv"
if [[ ! -e "$SUMMARY_PATH" ]]; then
  printf 'case\tdp\tsp\tep\tsp_policy\trequests\tseq_len\tstatus\tprofile_dir\tlog\n' > "$SUMMARY_PATH"
fi

run_case() {
  local case_name="$1"
  local dp="$2"
  local sp="$3"
  local sp_policy="$4"
  local case_dir="$OUTPUT_ROOT/$case_name"
  local profiler_dir="$case_dir/profile"
  local log_path="$case_dir/run.log"
  local command_path="$case_dir/command.txt"

  if [[ -e "$log_path" || -e "$profiler_dir" ]]; then
    printf 'Refusing to mix profile output in existing case directory: %s\n' "$case_dir" >&2
    exit 1
  fi
  mkdir -p "$case_dir" "$profiler_dir"

  local cmd=(
    python3 -u "$REPO_ROOT/examples/dummy_prefill.py"
    --num-seqs "$NUM_SEQS"
    --seq-len "$SEQ_LEN"
    --max-tokens "$MAX_TOKENS"
    --moe-routing-simulation-strategy "$MOE_ROUTING_SIMULATION_STRATEGY"
    --seed "$SEED"
    --max-num-seqs "$MAX_NUM_SEQS"
    --dp "$dp"
    --sp "$sp"
    --ep 16
    --model-path "$MODEL_PATH"
    --master-address "$MASTER_ADDRESS"
    --ray-address "$RAY_ADDRESS"
    --max-num-send-seqs "$MAX_NUM_SEND_SEQS"
    --max-num-recv-seqs "$MAX_NUM_RECV_SEQS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --loop-count "$LOOP_COUNT"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --enable-profiler
    --profiler-start-step "$PROFILER_START_STEP"
    --profiling-step "$PROFILING_STEP"
    --profiler-dir "$profiler_dir"
    --routing-strategy LeastBatch
    --sp-backend "$SP_BACKEND"
    --sp-size-policy "$sp_policy"
    --segment-size "$SEGMENT_SIZE"
    --long-request-sp-threshold "$LONG_REQUEST_SP_THRESHOLD"
    --enable-non-uniform-split
  )

  {
    printf 'SLIME_QP_NUM=%q' "$SLIME_QP_NUM"
    printf ' %q' "${cmd[@]}"
    printf '\n'
  } | tee "$command_path"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s\t%s\t%s\t16\t%s\t%s\t%s\tdry_run\t%s\t%s\n' \
      "$case_name" "$dp" "$sp" "$sp_policy" "$NUM_SEQS" "$SEQ_LEN" \
      "$profiler_dir" "$log_path" >> "$SUMMARY_PATH"
    return 0
  fi

  if "${cmd[@]}" 2>&1 | tee "$log_path"; then
    printf '%s\t%s\t%s\t16\t%s\t%s\t%s\tdone\t%s\t%s\n' \
      "$case_name" "$dp" "$sp" "$sp_policy" "$NUM_SEQS" "$SEQ_LEN" \
      "$profiler_dir" "$log_path" >> "$SUMMARY_PATH"
  else
    printf '%s\t%s\t%s\t16\t%s\t%s\t%s\tfailed\t%s\t%s\n' \
      "$case_name" "$dp" "$sp" "$sp_policy" "$NUM_SEQS" "$SEQ_LEN" \
      "$profiler_dir" "$log_path" >> "$SUMMARY_PATH"
    return 1
  fi
}

for case_name in "${cases[@]}"; do
  case "$case_name" in
    dp2sp8_ep16)
      run_case "$case_name" 2 8 long_short
      ;;
    dp16_ep16)
      run_case "$case_name" 16 1 legacy
      ;;
    *)
      printf 'Unknown case: %s (expected dp2sp8_ep16 or dp16_ep16)\n' "$case_name" >&2
      exit 1
      ;;
  esac
done

printf 'Profile matrix completed. Summary: %s\n' "$SUMMARY_PATH"
