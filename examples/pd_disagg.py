import os

from nanodeploy import LLM, SamplingParams
from nanodeploy.engine.sequence import Sequence

from transformers import AutoTokenizer


def main():
    path = os.path.expanduser(
        "/models/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/ba82a1060073fa0ecdc70d7b1922ec071f60cf3e"
    )
    tokenizer = AutoTokenizer.from_pretrained(path)

    prefill = LLM(
        path,
        enforce_eager=False,
        attention_dp=8,
        attention_sp=1,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
    )

    decode = LLM(
        path,
        enforce_eager=False,
        attention_dp=8,
        attention_sp=1,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
    )

    prefill_endpoints_info = prefill.p2p_init(
        decode.engine_id, decode.config.attn_world_size
    )
    decode_endpoints_info = decode.p2p_init(
        prefill.engine_id, prefill.config.attn_world_size
    )

    print(f"{prefill_endpoints_info}, {decode_endpoints_info}")


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
