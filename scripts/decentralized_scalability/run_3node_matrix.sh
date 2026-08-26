#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
RAY_ADDRESS="${RAY_ADDRESS:-10.102.252.174:6380}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/nanodeploy-decentralized-cpu-${RUN_TIMESTAMP}}"

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
    echo "usage: $0 [smoke|full]" >&2
    exit 2
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export CUDA_VISIBLE_DEVICES=""
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

echo "mode=${MODE}"
echo "ray_address=${RAY_ADDRESS}"
echo "output_root=${OUTPUT_ROOT}"

if [[ "${MODE}" == "smoke" ]]; then
    python3 -m scripts.decentralized_scalability.profile_router_cpu \
        --engine-counts 1,2,3 \
        --scaling-modes weak \
        --requests-per-engine 64 \
        --batch-sizes 8 \
        --prompt-lengths 64 \
        --repeats 2 \
        --output-dir "${OUTPUT_ROOT}/router"

    python3 -m scripts.decentralized_scalability.profile_frontend_ray_cpu \
        --ray-address "${RAY_ADDRESS}" \
        --node-counts 1,2,3 \
        --scaling-modes weak \
        --requests-per-engine 64 \
        --batch-sizes 8 \
        --prompt-lengths 64 \
        --ingress-repeats 2 \
        --event-warmup-iterations 2 \
        --event-iterations 10 \
        --output-dir "${OUTPUT_ROOT}/frontend"

    python3 -m scripts.decentralized_scalability.profile_local_scheduler_ray_cpu \
        --ray-address "${RAY_ADDRESS}" \
        --node-counts 1,2,3 \
        --scaling-modes weak \
        --batch-sizes 8 \
        --prompt-lengths 64 \
        --warmup-iterations 2 \
        --iterations 10 \
        --output-dir "${OUTPUT_ROOT}/scheduler"
else
    python3 -m scripts.decentralized_scalability.profile_router_cpu \
        --engine-counts 1,2,3 \
        --scaling-modes strong,weak \
        --strong-total-requests 4096 \
        --requests-per-engine 4096 \
        --batch-sizes 32,64,128 \
        --prompt-lengths 32,8000 \
        --repeats 5 \
        --output-dir "${OUTPUT_ROOT}/router"

    python3 -m scripts.decentralized_scalability.profile_frontend_ray_cpu \
        --ray-address "${RAY_ADDRESS}" \
        --node-counts 1,2,3 \
        --scaling-modes strong,weak \
        --strong-total-requests 4096 \
        --requests-per-engine 4096 \
        --batch-sizes 32,64,128 \
        --prompt-lengths 32,8000 \
        --ingress-repeats 5 \
        --event-warmup-iterations 20 \
        --event-iterations 500 \
        --output-dir "${OUTPUT_ROOT}/frontend"

    python3 -m scripts.decentralized_scalability.profile_local_scheduler_ray_cpu \
        --ray-address "${RAY_ADDRESS}" \
        --node-counts 1,2,3 \
        --scaling-modes strong,weak \
        --batch-sizes 32,64,128 \
        --prompt-lengths 32,8000 \
        --warmup-iterations 20 \
        --iterations 500 \
        --output-dir "${OUTPUT_ROOT}/scheduler"
fi

echo "completed output_root=${OUTPUT_ROOT}"
