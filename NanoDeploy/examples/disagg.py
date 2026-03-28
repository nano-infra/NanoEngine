"""Disaggregated (prefill-decode) LLM inference example.

Common config is set at top-level; per-role overrides use
--prefill.xxx / --decode.xxx scoping (overlay on top of common).

Usage:
    python disagg.py --model /models/deepseek-v3 \\
        --ray_address <node0-ip>:7078 \\
        --nanoctrl_address <node0-ip>:3000 \\
        --kvcache_block_size 64 \\
        --attention_dp 8 --ffn_ep 8 \\
        --prefill.master_address <node1-ip>:6006 \\
        --decode.master_address <node0-ip>:6006 \\
        --decode.loop_count 16

    python disagg.py --config disagg_config.yaml
"""

import os

import numpy as np
import ray
from jsonargparse import ActionConfigFile, ArgumentParser
from nanodeploy._cpp import deserialize, SamplingParams, Sequence
from nanodeploy.config import Config
from nanodeploy.llm_component import LLMComponent
from transformers import PreTrainedTokenizerFast


def main():
    parser = ArgumentParser(description="Disaggregated LLM inference example")
    parser.add_argument("--config", action=ActionConfigFile)

    # Common config (top-level): shared by both prefill & decode
    parser.add_class_arguments(Config, fail_untyped=False)

    # Per-role overrides: --prefill.xxx / --decode.xxx overlay on common
    parser.add_class_arguments(
        Config, nested_key="prefill", skip={"model"}, fail_untyped=False
    )
    parser.add_class_arguments(
        Config, nested_key="decode", skip={"model"}, fail_untyped=False
    )

    # Generation args
    parser.add_argument("--prompt", type=str, default="What is 1+1?")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)

    args = parser.parse_args()
    defaults = parser.get_defaults()

    # Build common config dict (excluding non-Config fields)
    extra_keys = {"config", "prompt", "max_tokens", "temperature", "prefill", "decode"}
    common = {k: v for k, v in vars(args).items() if k not in extra_keys}

    def build_config(ns, default_ns, mode: str) -> Config:
        """Merge common config with per-role overrides.

        Only values explicitly set by the user (differing from parser defaults)
        override the common config.
        """
        overrides = {
            k: v for k, v in vars(ns).items() if v != getattr(default_ns, k, v)
        }
        merged = {**common, **overrides, "mode": mode}
        return Config(**merged)

    prefill_config = build_config(args.prefill, defaults.prefill, "prefill")
    decode_config = build_config(args.decode, defaults.decode, "decode")

    # Launch engines
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
    prefill = LLMComponent.as_remote(prefill_config)
    decode = LLMComponent.as_remote(decode_config)

    print("\nEngines registered with NanoCtrl - automatic peer discovery enabled\n")

    # Build sequences
    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens, ignore_eos=False
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
