import argparse
import os
import time
from typing import Any

import ray
from transformers import AutoTokenizer

from nanodeploy import LLM, SamplingParams
from nanodeploy.engine.sequence import Sequence


DEFAULT_MODEL_PATH = (
    "/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-V3/snapshots/"
    "e815299b0bcbac849fa540c768ef21845365c9eb"
)
DEFAULT_SLIME_VISIBLE_DEVICES = ",".join(f"mlx5_{idx}" for idx in range(8))
PROXY_ENV_NAMES = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
)
RDMA_ENV_DEFAULTS = {
    "SLIME_VISIBLE_DEVICES": DEFAULT_SLIME_VISIBLE_DEVICES,
    "SLIME_GID_INDEX": "3",
    "SLIME_QP_NUM": "4",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one DeepSeek-V3 request through a two-node "
            "prefill/decode-disaggregated deployment."
        )
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--ray-address", default="10.102.252.174:6380")
    parser.add_argument(
        "--prefill-master-address",
        default="10.102.252.174:6006",
    )
    parser.add_argument(
        "--decode-master-address",
        default="10.102.243.60:6006",
    )
    parser.add_argument(
        "--prompt",
        default="请用三句话解释月亮为什么不会掉到地球上。",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1e-5)
    parser.add_argument("--decode-loop-count", type=int, default=1)
    parser.add_argument(
        "--decode-topology",
        choices=("dp8", "sp8", "bucket-sp8"),
        default="sp8",
        help=(
            "Decode attention topology: dp8 uses DP8/SP1 and isolates local "
            "decode/CUDA Graph; sp8 uses fixed DP1/SP8; bucket-sp8 uses "
            "DP1/SP8 with dynamic SP bucket scheduling."
        ),
    )
    parser.add_argument(
        "--dynamic-sp-bucket-policy",
        default="",
        help=(
            "Explicit scheduler policy for --decode-topology bucket-sp8, for "
            "example '1:1-127;5:128-383;6:384-639;7:640-895;"
            "8:896-4096'."
        ),
    )
    parser.add_argument(
        "--non-uniform-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable non-uniform KV placement for bucket-sp8 (default: "
            "enabled). Use --no-non-uniform-split only for comparison."
        ),
    )
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        help="Skip checkpoint loading and use initialized dummy weights.",
    )
    parser.add_argument(
        "--cuda-graph-mode",
        choices=("full", "piecewise"),
        default="full",
        help="Decode CUDA Graph mode (default: full).",
    )
    parser.add_argument(
        "--decode-eager",
        action="store_true",
        help="Disable decode CUDA Graph for debugging.",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--sp-backend",
        choices=("hao_basic", "nccl"),
        default="hao_basic",
    )
    parser.add_argument(
        "--optimize-decode-block-table",
        action="store_true",
        help=(
            "Enable decode RPC block-table filtering. It is disabled by "
            "default for the first correctness smoke."
        ),
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def decode_topology(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.decode_topology == "dp8":
        return 8, 1, 0
    if args.decode_topology == "sp8":
        return 1, 8, 8
    return 1, 8, 0


def decode_dynamic_sp_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.decode_topology != "bucket-sp8":
        return {
            "dynamic_sp_size_strategy": "legacy",
            "dynamic_sp_bucket_policy": "",
            "enable_non_uniform_split": False,
        }
    policy = args.dynamic_sp_bucket_policy.strip()
    if not policy:
        raise ValueError(
            "--dynamic-sp-bucket-policy is required with "
            "--decode-topology bucket-sp8"
        )
    return {
        "dynamic_sp_size_strategy": "bucket",
        "dynamic_sp_bucket_policy": policy,
        "enable_non_uniform_split": args.non_uniform_split,
    }


def configure_driver_environment() -> dict[str, str]:
    for name in PROXY_ENV_NAMES:
        os.environ.pop(name, None)
    for name, value in RDMA_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)
    return {name: os.environ[name] for name in RDMA_ENV_DEFAULTS}


def node_ip_from_master_address(address: str) -> str:
    raw = address.split("://", 1)[-1]
    return raw.rsplit(":", 1)[0]


def validate_ray_cluster(
    prefill_master_address: str,
    decode_master_address: str,
) -> None:
    alive_nodes = {
        node["NodeManagerAddress"]: node
        for node in ray.nodes()
        if node.get("Alive")
    }
    required_ips = {
        node_ip_from_master_address(prefill_master_address),
        node_ip_from_master_address(decode_master_address),
    }
    missing_ips = required_ips.difference(alive_nodes)
    if missing_ips:
        raise RuntimeError(
            "Ray cluster is missing required nodes: "
            + ", ".join(sorted(missing_ips))
        )
    for node_ip in sorted(required_ips):
        gpu_count = float(alive_nodes[node_ip].get("Resources", {}).get("GPU", 0))
        if gpu_count < 8:
            raise RuntimeError(
                f"Ray node {node_ip} exposes only {gpu_count:g} GPUs; 8 are required"
            )

    available_gpus = float(ray.available_resources().get("GPU", 0))
    if available_gpus < 16:
        raise RuntimeError(
            "The two-node smoke requires 16 currently available Ray GPUs, "
            f"but only {available_gpus:g} are available. Remove stale actors "
            "or placement groups before retrying."
        )
    print(
        "Ray cluster ready:",
        f"nodes={sorted(required_ips)}",
        f"available_gpus={available_gpus:g}",
        flush=True,
    )


def build_decode(args: argparse.Namespace) -> LLM:
    attention_dp, attention_sp, fixed_sp_size = decode_topology(args)
    dynamic_sp_kwargs = decode_dynamic_sp_kwargs(args)
    print(
        "Creating decode engine first:",
        args.decode_master_address,
        f"attention=DP{attention_dp}/SP{attention_sp}/TP1",
        "ffn=DP1/EP8/TP1",
        (
            "cuda_graph=disabled"
            if args.decode_eager
            else f"cuda_graph={args.cuda_graph_mode}"
        ),
        flush=True,
    )
    return LLM(
        args.model_path,
        enforce_eager=args.decode_eager,
        cuda_graph_mode=args.cuda_graph_mode,
        attention_dp=attention_dp,
        attention_sp=attention_sp,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
        mode="decode",
        scheduler_arch="legacy_global",
        master_address=args.decode_master_address,
        ray_address=args.ray_address,
        dummy_prefill=False,
        dummy_weight=args.dummy_weight,
        fixed_sp_size=fixed_sp_size,
        **dynamic_sp_kwargs,
        sp_backend=args.sp_backend,
        optimize_decode_block_table=args.optimize_decode_block_table,
        kvcache_block_size=64,
        loop_count=args.decode_loop_count,
        seed=0,
        max_num_seqs=args.max_num_seqs,
        max_num_recv_seqs=max(args.max_num_seqs, 16),
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


def build_prefill(args: argparse.Namespace) -> LLM:
    print(
        "Creating prefill engine second:",
        args.prefill_master_address,
        "attention=DP8/SP1/TP1",
        "ffn=DP1/EP8/TP1",
        flush=True,
    )
    return LLM(
        args.model_path,
        enforce_eager=True,
        attention_dp=8,
        attention_sp=1,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
        mode="prefill",
        scheduler_arch="legacy_global",
        master_address=args.prefill_master_address,
        ray_address=args.ray_address,
        dummy_prefill=False,
        dummy_weight=args.dummy_weight,
        sp_backend=args.sp_backend,
        kvcache_block_size=64,
        loop_count=1,
        seed=0,
        max_num_seqs=args.max_num_seqs,
        max_num_recv_seqs=max(args.max_num_seqs, 16),
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


def connect_kv_transfer(prefill: LLM, decode: LLM) -> None:
    print("Initializing P/D KV-transfer endpoints", flush=True)
    prefill_endpoints = prefill.p2p_init(
        decode.engine_id,
        decode.config.num_kvcache_blocks,
        decode.config.attn_world_size,
    )
    decode_endpoints = decode.p2p_init(
        prefill.engine_id,
        prefill.config.num_kvcache_blocks,
        prefill.config.attn_world_size,
    )
    prefill.p2p_connect(decode.engine_id, decode_endpoints)
    decode.p2p_connect(prefill.engine_id, prefill_endpoints)
    print("P/D KV-transfer endpoints connected", flush=True)


def make_sequence(
    tokenizer: Any,
    prompt: str,
    sampling_params: SamplingParams,
) -> Sequence:
    prompt_token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_token_ids = list(prompt_token_ids)
    if len(prompt_token_ids) < 8:
        raise ValueError(
            "The prompt must contain at least 8 tokens so fixed SP8 uses all ranks"
        )
    print(f"Prompt tokens: {len(prompt_token_ids)}", flush=True)
    return Sequence(prompt_token_ids, sampling_params=sampling_params)


def close_engine(engine: LLM | None, label: str) -> None:
    if engine is None:
        return
    try:
        engine.exit()
    except Exception as exc:
        print(f"Warning: failed to close {label} engine: {exc}", flush=True)


def report_completion_stage(
    tokenizer: Any,
    sequence: Sequence,
    stage: str,
    previous_count: int = 0,
) -> int:
    completion_token_ids = list(sequence.completion_token_ids)
    added_token_ids = completion_token_ids[previous_count:]
    print(f"{stage} total completion token IDs: {completion_token_ids}", flush=True)
    print(f"{stage} added token IDs: {added_token_ids}", flush=True)
    print(
        f"{stage} added text:",
        tokenizer.decode(added_token_ids, skip_special_tokens=True),
        flush=True,
    )
    return len(completion_token_ids)


def main() -> None:
    args = parse_args()
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.temperature <= 1e-10:
        raise ValueError(
            "NanoDeploy does not support temperature=0; use a small positive value"
        )
    if args.decode_loop_count <= 0:
        raise ValueError("--decode-loop-count must be positive")
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

    rdma_env = configure_driver_environment()
    print(f"RDMA environment: {rdma_env}", flush=True)
    ray.init(
        address=args.ray_address,
        ignore_reinit_error=True,
        runtime_env={"env_vars": rdma_env},
    )
    validate_ray_cluster(
        args.prefill_master_address,
        args.decode_master_address,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    sequence = make_sequence(tokenizer, args.prompt, sampling_params)

    decode: LLM | None = None
    prefill: LLM | None = None
    try:
        init_begin = time.perf_counter()
        decode = build_decode(args)
        prefill = build_prefill(args)
        print(
            f"Both engines initialized in {time.perf_counter() - init_begin:.2f}s",
            flush=True,
        )

        connect_kv_transfer(prefill, decode)

        prefill_begin = time.perf_counter()
        prefill.add_request(sequence)
        prefill.generate(use_tqdm=False)
        print(
            f"Prefill completed in {time.perf_counter() - prefill_begin:.3f}s",
            flush=True,
        )
        prefill_token_count = report_completion_stage(
            tokenizer,
            sequence,
            "Prefill",
        )

        decode_begin = time.perf_counter()
        decode.add_request(sequence)
        decode.generate(use_tqdm=False)
        print(
            f"KV migration and decode completed in "
            f"{time.perf_counter() - decode_begin:.3f}s",
            flush=True,
        )
        report_completion_stage(
            tokenizer,
            sequence,
            "Decode",
            previous_count=prefill_token_count,
        )
        prefill.free_to_be_migrated(sequence)

        completion_token_ids = list(sequence.completion_token_ids)
        if len(completion_token_ids) != args.max_tokens:
            raise RuntimeError(
                "Unexpected completion length: "
                f"expected {args.max_tokens}, got {len(completion_token_ids)}"
            )
        completion = tokenizer.decode(
            completion_token_ids,
            skip_special_tokens=True,
        )
        print("Completion token IDs:", completion_token_ids, flush=True)
        print("Completion:", completion, flush=True)
        print("DeepSeek-V3 one-request P/D smoke passed", flush=True)
    finally:
        close_engine(prefill, "prefill")
        close_engine(decode, "decode")
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
