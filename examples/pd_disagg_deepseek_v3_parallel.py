"""Run the DeepSeek-V3 P/D smoke with concurrent engine initialization.

This variant keeps the existing P/D topology and request flow, but builds the
prefill and decode engines in parallel.  Completion of both build futures is
the synchronization barrier before KV-transfer setup begins.
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import ray
from transformers import AutoTokenizer

from nanodeploy import LLM, SamplingParams

import pd_disagg_deepseek_v3 as serial_example


def build_decode(args: argparse.Namespace) -> LLM:
    print(
        "Creating decode engine concurrently:",
        args.decode_master_address,
        "attention=DP1/SP8/TP1",
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
        attention_dp=1,
        attention_sp=8,
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
        fixed_sp_size=8,
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
        "Creating prefill engine concurrently:",
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


def _timed_build(
    label: str,
    builder: Callable[[argparse.Namespace], LLM],
    args: argparse.Namespace,
) -> tuple[LLM, float]:
    begin = time.perf_counter()
    engine = builder(args)
    elapsed = time.perf_counter() - begin
    print(f"{label} engine initialized in {elapsed:.2f}s", flush=True)
    return engine, elapsed


def build_engines_parallel(
    args: argparse.Namespace,
) -> tuple[LLM, LLM]:
    wall_begin = time.perf_counter()
    engines: dict[str, LLM] = {}
    elapsed_by_label: dict[str, float] = {}
    errors: dict[str, Exception] = {}

    print("Starting concurrent decode and prefill initialization", flush=True)
    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="pd-engine-init",
    ) as pool:
        futures = {
            pool.submit(_timed_build, "decode", build_decode, args): "decode",
            pool.submit(_timed_build, "prefill", build_prefill, args): "prefill",
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                engine, elapsed = future.result()
            except Exception as exc:
                errors[label] = exc
            else:
                engines[label] = engine
                elapsed_by_label[label] = elapsed

    if errors:
        for label, engine in engines.items():
            serial_example.close_engine(engine, label)
        details = "; ".join(
            f"{label}: {type(exc).__name__}: {exc}"
            for label, exc in errors.items()
        )
        first_error = next(iter(errors.values()))
        raise RuntimeError(
            f"Concurrent P/D engine initialization failed ({details})"
        ) from first_error

    wall_elapsed = time.perf_counter() - wall_begin
    print(
        "Both engines initialized concurrently:",
        f"decode={elapsed_by_label['decode']:.2f}s",
        f"prefill={elapsed_by_label['prefill']:.2f}s",
        f"wall={wall_elapsed:.2f}s",
        flush=True,
    )
    return engines["decode"], engines["prefill"]


def main() -> None:
    args = serial_example.parse_args()
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

    rdma_env = serial_example.configure_driver_environment()
    print(f"RDMA environment: {rdma_env}", flush=True)
    ray.init(
        address=args.ray_address,
        ignore_reinit_error=True,
        runtime_env={"env_vars": rdma_env},
    )
    serial_example.validate_ray_cluster(
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
    sequence = serial_example.make_sequence(
        tokenizer,
        args.prompt,
        sampling_params,
    )

    decode: LLM | None = None
    prefill: LLM | None = None
    try:
        decode, prefill = build_engines_parallel(args)

        serial_example.connect_kv_transfer(prefill, decode)

        prefill_begin = time.perf_counter()
        prefill.add_request(sequence)
        prefill.generate(use_tqdm=False)
        print(
            f"Prefill completed in {time.perf_counter() - prefill_begin:.3f}s",
            flush=True,
        )

        decode_begin = time.perf_counter()
        decode.add_request(sequence)
        decode.generate(use_tqdm=False)
        print(
            "KV migration and decode completed in "
            f"{time.perf_counter() - decode_begin:.3f}s",
            flush=True,
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
        print("DeepSeek-V3 parallel-init P/D smoke passed", flush=True)
    finally:
        serial_example.close_engine(prefill, "prefill")
        serial_example.close_engine(decode, "decode")
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
