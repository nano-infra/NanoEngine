import os

from nanodeploy import LLM, SamplingParams
from nanodeploy.engine.sequence import Sequence

from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/models/model--Qwen--Qwen3-30B-A3B-FP8")
    tokenizer = AutoTokenizer.from_pretrained(path)

    decode = LLM(
        path,
        enforce_eager=False,
        attention_dp=1,
        attention_sp=8,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
        mode="decode",
        loop_count=16,
        master_address="10.103.5.41:6006",
        ray_address="10.103.5.41:7077",
        dummy_prefill=False,
        max_num_seqs=2,
        max_model_len=262144,
        max_num_batched_tokens=262144,
    )

    prefill = LLM(
        path,
        enforce_eager=True,
        loop_count=1,
        attention_dp=8,
        attention_sp=1,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
        mode="prefill",
        master_address="10.103.11.87:6006",
        ray_address="10.103.5.41:7077",
    )

    prefill_endpoints_info = prefill.p2p_init(
        decode.engine_id,
        decode.config.num_kvcache_blocks,
        decode.config.attn_world_size,
    )

    decode_endpoints_info = decode.p2p_init(
        prefill.engine_id,
        prefill.config.num_kvcache_blocks,
        prefill.config.attn_world_size,
    )

    prefill.p2p_connect(decode.engine_id, decode_endpoints_info)
    decode.p2p_connect(prefill.engine_id, prefill_endpoints_info)

    sampling_params = SamplingParams(temperature=0.1, max_tokens=512, ignore_eos=False)

    prompts = [
        """
As evening fell, dark clouds spread across the sky like spilled ink, gradually blotting out the light. Wind swirled with withered leaves outside the window, and soon, the patter of rain began—a gentle rhythm that grew denser by the minute, weaving a gray, misty net that enshrouded the entire town.
I leaned over my desk doing homework, the warm glow of the desk lamp diffusing a soft halo through the rain fog. From the living room came the rustle of Mom sorting vegetables, mixed with the faint sound of Dad flipping through the newspaper and occasional news broadcasts from the TV. These subtle noises, set against the rain, brought an overwhelming sense of peace. Suddenly, the desk lamp flickered twice, and the room was plunged into darkness. "Power outage?" I exclaimed instinctively, fumbling for my phone in a fluster.
"Don’t panic—I’ll find candles," Dad’s voice came out of the dark, steady and reassuring. A lighter flame flared to life, casting light on his familiar silhouette. Mom approached holding candles, their glow illuminating the fine lines at the corners of her eyes and the freshly cut fruit on the table. "Put away your homework quickly; it’s too dark for your eyes," she said, placing a candle on the desk corner before turning to the kitchen. "I just made soup—tonight we’ll chat over candlelight while we eat."
The candle flames danced, throwing our shadows onto the wall, now bright, now dim. Rain tapped against the windowpanes like nature’s accompaniment. Dad told stories of power outages from his childhood, how the kids would gather around candles to play games, while Mom chatted about remembering to buy backup lightbulbs tomorrow. I held the warm soup bowl in my hands, listening to my family’s voices, the aroma of the soup and the faint scent of candle wax lingering in my nose, filling my heart with warmth.
On usual days, life felt so hurried—rushing through studies, chasing after time—that I rarely had moments like this to sit quietly with my family. This unexpected power outage was like pressing pause on life, letting me feel the most genuine warmth in ordinary days. It turned out happiness didn’t need to be earth-shaking; it might just be a rainy night like this, a candle’s glow, a bowl of hot soup, and sitting around with family, chatting about trivial matters.
Before long, the lights suddenly came back on, flooding the room with brightness. The rain was still falling, but my heart was brimming with warmth. I looked out at the world, refreshed by the rain, and suddenly understood—life’s beauties are often hidden in these unplanned little interludes. As long as we pay attention, we’ll find that warmth is always there.
Score this composition.
"""
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

    for prompt, seq in zip(prompts, seqs):
        token_ids = seq.completion_token_ids
        output = {"text": tokenizer.decode(token_ids), "token_ids": token_ids}
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
