import os
from nanodeploy import LLM
from nanovllm import SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/models/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/ba82a1060073fa0ecdc70d7b1922ec071f60cf3e")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        enforce_eager=False,
        tensor_parallel_size=1,
        expert_parallel_size=8,
        data_parallel_size=8
    )

    sampling_params = SamplingParams(temperature=0.1, max_tokens=128, ignore_eos=True)
    prompts = [
        "你好",
        "how to bake a chocolate cake from scratch",
        "what are the benefits of meditation",
        "list 5 famous scientists and their contributions",
        "explain quantum computing in simple terms",
        "how to bake a chocolate cake from scratch",
        "what are the benefits of meditation",
        "list 5 famous scientists and their contributions",
    ]
    prompts = [prompt for prompt in prompts]
    prompts = prompts[0:512]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
