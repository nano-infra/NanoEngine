"""Non-disaggregated LLM inference example.

Usage:
    python non_disagg.py --model /models/deepseek-v3 --loop_count 1 --kvcache_block_size 64
    python non_disagg.py --config config.yaml
"""

import os

from jsonargparse import ActionConfigFile, ArgumentParser
from nanodeploy import Sequence
from nanodeploy.config import Config
from nanodeploy.llm_component import LLM
from nanodeploy.sampling_params import SamplingParams
from transformers import PreTrainedTokenizerFast


def main():
    parser = ArgumentParser(description="Non-disaggregated LLM inference example")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_class_arguments(Config, fail_untyped=False)
    parser.add_argument("--prompt", type=str, default="What is 1+1?")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
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
        ignore_eos=False,
        temperature=args.temperature,
    )
    prompts = [args.prompt]

    seqs = [
        Sequence(
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ),
            sampling_params=sampling_params,
        )
        for p in prompts
    ]
    llm.add_request(seqs)
    llm.generate()

    for prompt, seq in zip(prompts, seqs):
        token_ids = seq.completion_token_ids
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {llm.tokenizer.decode(token_ids)!r}")


if __name__ == "__main__":
    main()
