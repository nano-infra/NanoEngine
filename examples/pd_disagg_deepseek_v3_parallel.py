"""Run the DeepSeek-V3 P/D smoke with concurrent engine initialization.

This variant keeps the existing P/D topology and request flow, but builds the
prefill and decode engines in parallel.  Completion of both build futures is
the synchronization barrier before KV-transfer setup begins.
"""

import argparse
import itertools
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import ray
from transformers import AutoTokenizer

from nanodeploy import LLM, SamplingParams
from nanodeploy.engine.sequence import Sequence

import pd_disagg_deepseek_v3 as serial_example


DEFAULT_BATCH_PROMPTS = (
    "请用三句话解释月亮为什么不会掉到地球上。",
    "请说明彩虹是怎样形成的，并解释为什么通常能看到多种颜色。",
    "如果一个水杯装满冰水，杯子外壁为什么会出现水珠？",
    "请比较太阳能和风能各自的主要优点与局限。",
    "为什么人在高海拔地区更容易感到呼吸困难？",
    "请用一个简单的生活例子解释什么是机会成本。",
    "请给出三个提高 Python 程序运行效率的通用方法。",
    "假设你要设计一个可靠的分布式服务，应优先考虑哪些故障场景？",
)

def parse_positive_int_csv(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of integers"
        ) from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("all token lengths must be positive")
    return result


def parse_args() -> argparse.Namespace:
    parser = serial_example.build_arg_parser()
    parser.description = (
        "Run concurrent DeepSeek-V3 requests through a two-node "
        "prefill/decode-disaggregated deployment with parallel engine "
        "initialization."
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1,
        help=(
            "Number of requests to enqueue together. Batches use distinct "
            "built-in prompts (default: 1)."
        ),
    )
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Continue decoding after EOS until --max-tokens. By default EOS "
            "stops each request."
        ),
    )
    parser.add_argument(
        "--prompt-token-lengths",
        type=parse_positive_int_csv,
        default=(),
        help=(
            "Optional exact prompt-token lengths, one per request. Padding is "
            "inserted before the assistant marker. The short five-bucket "
            "smoke uses 1024,768,512,256,64."
        ),
    )
    return parser.parse_args()


def select_prompts(args: argparse.Namespace) -> list[str]:
    if args.num_requests == 1:
        return [args.prompt]
    if args.num_requests > len(DEFAULT_BATCH_PROMPTS):
        raise ValueError(
            "--num-requests exceeds the number of built-in batch prompts: "
            f"{args.num_requests} > {len(DEFAULT_BATCH_PROMPTS)}"
        )
    return list(DEFAULT_BATCH_PROMPTS[: args.num_requests])


def build_decode(args: argparse.Namespace) -> LLM:
    attention_dp, attention_sp, fixed_sp_size = serial_example.decode_topology(args)
    dynamic_sp_kwargs = serial_example.decode_dynamic_sp_kwargs(args)
    print(
        "Creating decode engine concurrently:",
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


def make_sequence_with_token_length(
    tokenizer,
    prompt: str,
    sampling_params: SamplingParams,
    target_length: int,
) -> Sequence:
    prompt_token_ids = list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    if target_length < len(prompt_token_ids):
        raise ValueError(
            "--prompt-token-lengths entry is shorter than its tokenized "
            f"prompt: target={target_length}, actual={len(prompt_token_ids)}"
        )

    missing = target_length - len(prompt_token_ids)
    filler_token_ids = tokenizer.encode("\n", add_special_tokens=False)
    if not filler_token_ids:
        raise RuntimeError("Tokenizer produced no token for the prompt filler")
    filler = list(itertools.islice(itertools.cycle(filler_token_ids), missing))
    # DeepSeek's final chat-template token is the assistant marker. Keep all
    # special-token ordering intact and place harmless newlines immediately
    # before it.
    prompt_token_ids[-1:-1] = filler
    if len(prompt_token_ids) != target_length:
        raise AssertionError("Failed to construct the requested prompt length")
    print(f"Prompt tokens: {len(prompt_token_ids)}", flush=True)
    return Sequence(prompt_token_ids, sampling_params=sampling_params)


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
    args = parse_args()
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.num_requests <= 0:
        raise ValueError("--num-requests must be positive")
    if args.num_requests > args.max_num_seqs:
        raise ValueError(
            "--num-requests cannot exceed --max-num-seqs: "
            f"{args.num_requests} > {args.max_num_seqs}"
        )
    if args.prompt_token_lengths and (
        len(args.prompt_token_lengths) != args.num_requests
    ):
        raise ValueError(
            "--prompt-token-lengths must contain exactly --num-requests "
            f"entries: got {len(args.prompt_token_lengths)} lengths for "
            f"{args.num_requests} requests"
        )
    if (
        args.prompt_token_lengths
        and max(args.prompt_token_lengths) > args.max_model_len
    ):
        raise ValueError(
            "--prompt-token-lengths cannot exceed --max-model-len: "
            f"max={max(args.prompt_token_lengths)}, "
            f"max_model_len={args.max_model_len}"
        )
    serial_example.decode_dynamic_sp_kwargs(args)
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
        ignore_eos=args.ignore_eos,
    )
    prompts = select_prompts(args)
    for request_index, prompt in enumerate(prompts):
        print(f"Request[{request_index}] prompt: {prompt}", flush=True)
    if args.prompt_token_lengths:
        sequences = [
            make_sequence_with_token_length(
                tokenizer,
                prompt,
                sampling_params,
                target_length,
            )
            for prompt, target_length in zip(
                prompts,
                args.prompt_token_lengths,
                strict=True,
            )
        ]
    else:
        sequences = [
            serial_example.make_sequence(
                tokenizer,
                prompt,
                sampling_params,
            )
            for prompt in prompts
        ]
    print(
        f"Submitting {len(sequences)} requests in one batch",
        flush=True,
    )

    decode: LLM | None = None
    prefill: LLM | None = None
    try:
        decode, prefill = build_engines_parallel(args)

        serial_example.connect_kv_transfer(prefill, decode)

        prefill_begin = time.perf_counter()
        prefill.add_request(sequences)
        prefill.generate(use_tqdm=False)
        print(
            f"Prefill completed in {time.perf_counter() - prefill_begin:.3f}s",
            flush=True,
        )
        prefill_token_counts = [
            serial_example.report_completion_stage(
                tokenizer,
                sequence,
                f"Prefill request[{request_index}]",
            )
            for request_index, sequence in enumerate(sequences)
        ]

        decode_begin = time.perf_counter()
        decode.add_request(sequences)
        decode.generate(use_tqdm=False)
        print(
            "KV migration and decode completed in "
            f"{time.perf_counter() - decode_begin:.3f}s",
            flush=True,
        )
        for request_index, (sequence, prefill_token_count) in enumerate(
            zip(sequences, prefill_token_counts, strict=True)
        ):
            serial_example.report_completion_stage(
                tokenizer,
                sequence,
                f"Decode request[{request_index}]",
                previous_count=prefill_token_count,
            )
        prefill.free_to_be_migrated(sequences)

        for request_index, (prompt, sequence) in enumerate(
            zip(prompts, sequences, strict=True)
        ):
            completion_token_ids = list(sequence.completion_token_ids)
            completion_length = len(completion_token_ids)
            if not 0 < completion_length <= args.max_tokens:
                raise RuntimeError(
                    f"Request[{request_index}] invalid completion length: "
                    f"expected 1..{args.max_tokens}, got {completion_length}"
                )
            if args.ignore_eos and completion_length != args.max_tokens:
                raise RuntimeError(
                    f"Request[{request_index}] completion length mismatch "
                    "with --ignore-eos: "
                    f"expected {args.max_tokens}, got {completion_length}"
                )
            completion = tokenizer.decode(
                completion_token_ids,
                skip_special_tokens=True,
            )
            print(
                f"Request[{request_index}] prompt:",
                prompt,
                flush=True,
            )
            print(
                f"Request[{request_index}] completion token IDs:",
                completion_token_ids,
                flush=True,
            )
            print(
                f"Request[{request_index}] completion:",
                completion,
                flush=True,
            )
        print(
            "DeepSeek-V3 parallel-init P/D smoke passed:",
            f"requests={len(sequences)}",
            f"max_tokens_per_request={args.max_tokens}",
            "completion_lengths="
            f"{[len(sequence.completion_token_ids) for sequence in sequences]}",
            flush=True,
        )
    finally:
        serial_example.close_engine(prefill, "prefill")
        serial_example.close_engine(decode, "decode")
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
