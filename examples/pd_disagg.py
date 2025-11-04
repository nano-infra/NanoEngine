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
    )

    prefill = LLM(
        path,
        enforce_eager=False,
        attention_dp=8,
        attention_sp=1,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
        mode="prefill",
        master_address="127.0.0.1:6006",
    )

    prefill_endpoints_info = prefill.p2p_init(
        decode.engine_id, decode.config.attn_world_size
    )
    decode_endpoints_info = decode.p2p_init(
        prefill.engine_id, prefill.config.attn_world_size
    )

    print(f"{prefill_endpoints_info}, {decode_endpoints_info}")

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

    prefill.add_request(seqs)
    prefill.generate()

    decode.add_request(seqs)
    decode.generate()

    prefill.free_to_be_migrated(seqs)

    print([(seq.seq_id, seq.status) for seq in seqs])
    print(
        len(prefill.scheduler.block_manager(0).free_block_ids),
        len(prefill.scheduler.block_manager(0).blocks),
    )

    for prompt, seq in zip(prompts, seqs):
        token_ids = seq.completion_token_ids
        output = {"text": tokenizer.decode(token_ids), "token_ids": token_ids}
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
