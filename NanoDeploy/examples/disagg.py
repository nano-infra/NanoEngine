"""Disaggregated (prefill-decode) LLM inference example.

Shared fields (model, ray_address, nanoctrl_address) are specified once;
per-role overrides use --prefill.xxx / --decode.xxx scoping.

RAY_ADDRESS and NANOCTRL_ADDRESS are read from environment variables.

Usage:
    python disagg.py --model /models/deepseek-v3 \\
        --prefill.master_address 10.102.97.183:6006 \\
        --decode.master_address 10.102.97.179:6006 \\
        --decode.loop_count 16

    python disagg.py --config disagg_config.yaml
"""

import os

import numpy as np
import ray
from jsonargparse import ActionConfigFile, ArgumentParser
from nanodeploy._cpp import deserialize
from nanodeploy.config import Config
from nanodeploy.engine.sequence import Sequence
from nanodeploy.llm_component import LLMComponent
from nanodeploy.sampling_params import SamplingParams
from transformers import AutoTokenizer

# Fields that are shared across prefill/decode and should not appear in scoped groups.
_SHARED_FIELDS = {"model", "ray_address", "nanoctrl_address"}


def main():
    parser = ArgumentParser(description="Disaggregated LLM inference example")
    parser.add_argument("--config", action=ActionConfigFile)

    # Shared args (top-level, written once)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model directory (shared by prefill & decode)",
    )

    # Per-role scoped args: --prefill.xxx / --decode.xxx
    parser.add_class_arguments(
        Config, nested_key="prefill", skip=_SHARED_FIELDS, fail_untyped=False
    )
    parser.add_class_arguments(
        Config, nested_key="decode", skip=_SHARED_FIELDS, fail_untyped=False
    )

    # Generation args
    parser.add_argument("--prompt", type=str, default="What is 1+1?")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)

    args = parser.parse_args()

    # Read shared addresses from env vars
    ray_address = os.environ.get("RAY_ADDRESS", "127.0.0.1:6379")
    nanoctrl_address = os.environ.get("NANOCTRL_ADDRESS")

    # Build per-role Configs by injecting shared fields
    extra = ("config", "prompt", "max_tokens", "temperature")

    def build_config(ns, mode: str) -> Config:
        role_args = {k: v for k, v in vars(ns).items() if k not in extra}
        role_args.update(
            model=args.model,
            ray_address=ray_address,
            nanoctrl_address=nanoctrl_address,
            mode=mode,
        )
        return Config(**role_args)

    prefill_config = build_config(args.prefill, "prefill")
    decode_config = build_config(args.decode, "decode")

    # Launch engines
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prefill = LLMComponent.as_remote(prefill_config)
    decode = LLMComponent.as_remote(decode_config)

    print("\nEngines registered with NanoCtrl - automatic peer discovery enabled\n")

    # Build prompts & sequences
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=False,
    )
    seqs = [
        Sequence(
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": args.prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ),
            sampling_params=sampling_params,
        )
    ]

    # --- Prefill ---
    ray.get(prefill.add_request.remote(seqs))
    serialized_seqs = ray.get(prefill.generate.remote(return_serialized=True))

    print(f"\nPrefill returned {len(serialized_seqs)} serialized sequences.")

    migrated_seqs = []
    for i, blob in enumerate(serialized_seqs):
        buf = np.frombuffer(blob, dtype=np.uint8)
        deserialized = deserialize(buf.ctypes.data, len(buf))
        migrated_seqs.extend(deserialized)
        print(f"  [{i}] Deserialized {len(deserialized)} seq(s) from {len(blob)} bytes")

    if not migrated_seqs:
        print("No sequences migrated from prefill. Exiting.")
        return

    # --- Decode ---
    ray.get(decode.add_request.remote(migrated_seqs))
    finished_seqs = ray.get(decode.generate.remote())

    # Free migrated sequences in prefill engine
    ray.get(prefill.free_to_be_migrated.remote(migrated_seqs))

    # Print results
    for seq in finished_seqs:
        token_ids = seq.completion_token_ids
        print(f"\nSeq ID: {seq.seq_id}")
        print(f"Completion: {tokenizer.decode(token_ids)!r}")


if __name__ == "__main__":
    main()
