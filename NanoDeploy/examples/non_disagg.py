"""Non-disaggregated LLM inference example (GPU and Ascend NPU).

Backend selection (priority order):
  1. --backend_type / --device_type CLI flags
  2. NANO_BACKEND / NANO_DEVICE_TYPE environment variables
  3. Hardware auto-detection (torch_npu → Hopper → gpu_generic)

GPU (CUDA) usage:
    python non_disagg.py \\
        --ray_address 10.102.97.179:7078 --master_address 10.102.97.179:6006 \\
        --model /models/Qwen3-30B-A3B-FP8 \\
        --attention_dp 8 --ffn_ep 8 --kvcache_block_size 256 \\
        --device_type cuda

Ascend NPU usage:
    python non_disagg.py \\
        --ray_address 10.102.97.179:7078 --master_address 10.102.97.179:6006 \\
        --model /models/Qwen3-235B-A22B \\
        --attention_dp 8 --ffn_ep 8 --kvcache_block_size 256 \\
        --device_type npu --backend_type ascend --enforce_eager true

Config file usage:
    python non_disagg.py --config config.yaml
"""

import os

from jsonargparse import ActionConfigFile, ArgumentParser
from nanodeploy.config import Config
from nanodeploy.engine.sequence import Sequence
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
