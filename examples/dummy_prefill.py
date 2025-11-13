import os

from nanodeploy import LLM, SamplingParams
from nanodeploy.engine.sequence import Sequence

from transformers import AutoTokenizer


def main():
    path = os.path.expanduser(
        "/models/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/ba82a1060073fa0ecdc70d7b1922ec071f60cf3e"
    )
    tokenizer = AutoTokenizer.from_pretrained(path)

    decode = LLM(
        path,
        enforce_eager=False,
        attention_dp=8,
        attention_sp=1,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
        mode="decode",
        master_address="127.0.0.1:6006",
        ray_address="10.103.5.41:7077",
        dummy_prefill=True,
        dummy_weight=True,
        perfect_eplb=True,
        max_num_seqs=2,
        max_model_len=524288,
        max_num_batched_tokens=524288,
        loop_count=16,
    )

    sampling_params = SamplingParams(temperature=0.1, max_tokens=256, ignore_eos=True)
    # prompts = [
    #     "你好",
    #     "how to bake a chocolate cake from scratch",
    #     "what are the benefits of meditation",
    #     "list 5 famous scientists and their contributions",
    #     "explain quantum computing in simple terms",
    #     "how to bake a chocolate cake from scratch",
    #     "what are the benefits of meditation",
    #     "list 5 famous scientists and their contributions",
    # ]

    seqs = [
        Sequence(
            [0] * 400000,
            sampling_params=sampling_params,
        )
    ]

    decode.add_request(seqs)
    decode.generate()

    for seq in seqs:
        token_ids = seq.completion_token_ids
        output = {"text": tokenizer.decode(token_ids), "token_ids": token_ids}
        print(output)


if __name__ == "__main__":
    main()
