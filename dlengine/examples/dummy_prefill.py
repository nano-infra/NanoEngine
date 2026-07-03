import os

import numpy as np
from dlengine.llm_component import LLM
from dlengine.sampling_params import SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/models/qwen3-235B-Instruct-2507-FP8")

    from dlengine.config import Config

    config = Config(
        model=path,
        enforce_eager=False,
        attention_dp=8,
        attention_sp=1,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
        mode="decode",
        master_address="10.102.97.179:26444",
        ray_address="10.102.97.179:7078",
        dummy_prefill=True,
        dummy_weight=True,
        perfect_eplb=True,
        max_num_seqs=128,
        max_model_len=200_000,
        max_num_batched_tokens=200_000,
        # max_num_send_seqs=128, # Not standard config args, checking if Config supports them or if they go into kwargs of Config. Config definition has them.
        # max_num_recv_seqs=130,
        # kvcache_block_size=256,
        # enable_profiler=False,
        max_num_send_seqs=128,
        max_num_recv_seqs=130,
        kvcache_block_size=256,
        enable_profiler=False,
        log_level="INFO",
    )
    decode = LLM(config)

    sampling_params = SamplingParams(temperature=0.1, max_tokens=256, ignore_eos=True)

    long_prompts = [
        np.random.randint(0, 10001, size=400000).tolist(),
        # [0] * 400000,
    ]
    long_prompts = []

    short_prompts = [
        np.random.randint(0, 10001, size=2048).tolist()
        # [0] * 2048,
        for _ in range(512)
    ]

    for prompt in long_prompts + short_prompts:
        decode.add_request(prompt, sampling_params=sampling_params)
    decode.generate()


if __name__ == "__main__":
    main()
