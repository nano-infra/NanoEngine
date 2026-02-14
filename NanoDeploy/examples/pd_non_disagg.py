import os

from nanodeploy.config import Config
from nanodeploy.engine.sequence import Sequence

from nanodeploy.llm_component import LLM
from nanodeploy.sampling_params import SamplingParams
from transformers import AutoTokenizer


def main():

    path = os.path.expanduser("/models/qwen3-235B-Instruct-2507-FP8")
    tokenizer = AutoTokenizer.from_pretrained(path)
    config = Config(
        model=path,
        enforce_eager=False,
        attention_dp=8,
        attention_sp=1,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
        ray_address="10.102.97.179:7078",
        master_address="10.102.97.179:29901",
        loop_count=16,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.9,
        log_level="INFO",
        kvcache_block_size=256,
    )
    llm = LLM(config)

    sampling_params = SamplingParams(max_tokens=1024)
    prompts = [
        "Help me write a script for oh-my-zsh install configuration.",
    ]

    seqs = [
        Sequence(
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ),
            sampling_params=sampling_params,
        )
        for prompt in prompts
    ]
    llm.add_request(seqs)
    llm.generate()

    for prompt, seq in zip(prompts, seqs):
        token_ids = seq.completion_token_ids
        output = {"text": llm.tokenizer.decode(token_ids), "token_ids": token_ids}
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
