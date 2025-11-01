import os

from nanodeploy import LLM, SamplingParams
from nanodeploy.engine.sequence import Sequence

from transformers import AutoTokenizer


def main():
    path = os.path.expanduser(
        "/models/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/ba82a1060073fa0ecdc70d7b1922ec071f60cf3e"
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        enforce_eager=True,
        attention_dp=2,
        attention_sp=1,
        attention_tp=4,
        ffn_dp=2,
        ffn_ep=1,
        ffn_tp=4,
    )

    sampling_params = SamplingParams(temperature=0.1, max_tokens=128, ignore_eos=False)
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

    seqs = [
        Sequence(
                llm.tokenizer.encode(
                    llm.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
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
