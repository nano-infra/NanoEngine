import os
from nanodeploy import LLM
from nanovllm import SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/models/models--deepseek-ai--DeepSeek-V3/snapshots/4c1f24cc10a2a1894304c7ab52edd9710c047571/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        expert_parallel_size=8,
        data_parallel_size=8
    )

    sampling_params = SamplingParams(temperature=0.1, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
