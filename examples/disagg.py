"""Disaggregated (prefill-decode) LLM inference example.

Common config is set at top-level; per-role overrides use
--prefill.xxx / --decode.xxx scoping (overlay on top of common).

Usage:
    python disagg.py --model /models/deepseek-v3 \\
        --ray_address auto \\
        --ctrl_address <node0-ip>:4479 \\
        --kvcache_block_size 64 \\
        --attention_dp 8 --ffn_ep 8

    python disagg.py --config disagg_config.yaml
"""

import os
import sys
import uuid

import ray
from dlengine._rust.proto import RequestIn, RequestMigrate, SamplingParams
from dlengine.config import Config
from dlengine.llm_component import LLMComponent
from jsonargparse import ActionConfigFile, ArgumentParser
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
    parser.add_argument(
        "--dsv4_encoding_dir",
        type=str,
        default=None,
        help="Path to encoding_dsv4.py (DSv4 model has no chat_template).",
    )

    args = parser.parse_args()
    defaults = parser.get_defaults()

    # Build common config dict (excluding non-Config fields)
    extra_keys = {
        "config",
        "prompt",
        "max_tokens",
        "temperature",
        "prefill",
        "decode",
        "dsv4_encoding_dir",
    }
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

    # DSv4-FP8-SGlang has no chat_template — fall back to the optional
    # encoding_dsv4 script when --dsv4_encoding_dir is provided.
    dsv4_encode_messages = None
    if args.dsv4_encoding_dir:
        sys.path.insert(0, os.path.abspath(args.dsv4_encoding_dir))
        from encoding_dsv4 import (  # type: ignore
            encode_messages as dsv4_encode_messages,
        )

    def encode_prompt(prompt: str) -> list[int]:
        if dsv4_encode_messages is not None:
            text = dsv4_encode_messages(
                [{"role": "user", "content": prompt}],
                thinking_mode="chat",
            )
        elif getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt
        return tokenizer.encode(text)

    prompt_token_ids = encode_prompt(args.prompt)
    seq_id = uuid.uuid4().int & ((1 << 63) - 1)
    request_payload = RequestIn(
        seq_id,
        prompt_token_ids,
        sampling_params,
        0,
    ).to_bytes()

    # --- Prefill ---
    ray.get(prefill.add_request_payload.remote(request_payload))
    migration_payloads = ray.get(prefill.generate.remote(return_serialized=True))

    print(f"\nPrefill returned {len(migration_payloads)} migration payloads.")

    migrated_seq_ids = []
    for i, blob in enumerate(migration_payloads):
        seq_id, first_token = RequestMigrate.from_bytes(blob).metadata
        migrated_seq_ids.append(int(seq_id))
        print(
            f"  [{i}] seq_id={seq_id}, first_token={first_token}, payload={len(blob)} bytes"
        )

    if not migration_payloads:
        print("No sequences migrated from prefill. Exiting.")
        return

    # --- Decode ---
    for payload in migration_payloads:
        ray.get(decode.add_request_payload.remote(payload))
    outputs = ray.get(decode.generate.remote())

    # Free migrated sequences in prefill engine
    ray.get(prefill.free_to_be_migrated_ids.remote(migrated_seq_ids))

    # Print results
    for output in outputs:
        token_ids = output["token_ids"]
        print(f"\nSeq ID: {output['seq_id']}")
        print(f"Completion: {tokenizer.decode(token_ids)!r}")


if __name__ == "__main__":
    main()
