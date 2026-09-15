"""Non-disaggregated LLM inference example.

Usage:
    python non_disagg.py --model /models/deepseek-v3 --kvcache_block_size 64
    python non_disagg.py --config config.yaml
"""

import os
import sys
import uuid

from dlengine._rust.proto import RequestIn, SamplingParams
from dlengine.config import Config
from dlengine.engine.llm_component import LLM
from dlengine.offline import generate
from jsonargparse import ActionConfigFile, ArgumentParser
from transformers import PreTrainedTokenizerFast


def main():
    parser = ArgumentParser(description="Non-disaggregated LLM inference example")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_class_arguments(Config, fail_untyped=False)
    parser.add_argument("--prompt", type=str, default="What is 1+1?")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--ignore_eos", action="store_true")
    parser.add_argument("--dsv4_encoding_dir", type=str, default=None)
    args = parser.parse_args()

    # Build Config from parsed args (exclude extra args)
    config_args = {
        k: v
        for k, v in vars(args).items()
        if k not in ("config", "prompt", "max_tokens", "temperature")
    }
    config = Config(**config_args)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.model)
    llm = LLM(config)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        temperature=args.temperature,
    )
    prompts = [args.prompt]
    dsv4_encode_messages = None
    if args.dsv4_encoding_dir:
        sys.path.insert(0, os.path.abspath(args.dsv4_encoding_dir))
        from encoding_dsv4 import encode_messages as dsv4_encode_messages

    def encode_prompt(prompt: str) -> list[int]:
        if dsv4_encode_messages is not None:
            prompt = dsv4_encode_messages(
                [{"role": "user", "content": prompt}],
                thinking_mode="chat",
            )
        elif getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return tokenizer.encode(prompt)

    seq_ids = []
    for prompt in prompts:
        seq_id = uuid.uuid4().int & ((1 << 63) - 1)
        payload = RequestIn(
            seq_id,
            encode_prompt(prompt),
            sampling_params,
            0,
        ).to_bytes()
        llm.add_request_payload(payload)
        seq_ids.append(seq_id)
    outputs = {out["seq_id"]: out for out in generate(llm)}

    for prompt, seq_id in zip(prompts, seq_ids):
        token_ids = outputs.get(seq_id, {}).get("token_ids", [])
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {llm.tokenizer.decode(token_ids)!r}")


if __name__ == "__main__":
    main()
