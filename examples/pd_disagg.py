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
        attention_dp=1,
        attention_sp=8,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
        mode="decode",
        loop_count=1,
        master_address="10.103.5.41:6006",
        ray_address="10.103.5.41:7077",
        dummy_prefill=False,
        max_num_seqs=2,
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

    sampling_params = SamplingParams(temperature=0.1, max_tokens=128, ignore_eos=False)
    prompts = [
        """
    雨夜的暖光​
    傍晚时分，乌云像被打翻的墨汁，渐渐染黑了整片天空。风卷着枯叶在窗外打着旋，不一会儿，淅淅沥沥的雨声便敲起了节拍，由疏到密，最终织成一张灰蒙蒙的雨网，将整个小城笼罩其中。​
    我趴在书桌前写作业，台灯的暖光在雨雾中晕开一圈柔和的光晕。客厅里传来妈妈择菜的沙沙声，夹杂着爸爸翻报纸的轻响，偶尔还有电视里新闻播报的声音，这些细碎的声响在雨声的衬托下，格外让人安心。忽然，台灯闪烁了两下，屋里瞬间陷入黑暗。“停电了？” 我下意识地喊了一声，手忙脚乱地摸索着手机。​
    “别慌，我找蜡烛。” 爸爸的声音从黑暗中传来，带着沉稳的力量。打火机的火苗亮起，映出他熟悉的轮廓。妈妈拿着蜡烛走过来，烛光照亮了她眼角的细纹，也照亮了桌上刚切好的水果。“快把作业收一收，这么暗伤眼睛。” 她把蜡烛放在桌角，转身去厨房忙活，“正好炖了汤，咱们今晚就着烛光喝汤聊天。”​
    烛光摇曳，将我们的影子投在墙上，忽明忽暗。雨点敲打着窗户，像是大自然的伴奏。爸爸说起他小时候停电的趣事，说那时候孩子们会围着蜡烛做游戏，妈妈则念叨着明天要记得买备用灯泡。我捧着温热的汤碗，听着家人的话语，鼻尖萦绕着汤的鲜香和蜡烛淡淡的蜡油味，心里暖洋洋的。​
    平日里总觉得生活匆匆忙忙，忙着学习，忙着追赶时间，却很少有这样静下心来和家人相处的时刻。这场突如其来的停电，像是按下了生活的暂停键，让我感受到了平凡日子里最真切的温暖。原来幸福不必惊天动地，也许就是这样一个雨夜，一盏烛光，一碗热汤，和家人围坐在一起，聊着无关紧要的琐事。​
    不知过了多久，灯光突然亮起，屋里恢复了明亮。雨还在下，但我的心里却充满了暖意。我看着窗外被雨水冲刷得格外清新的世界，忽然明白，生活中的美好，往往就藏在这些不期而遇的小插曲里。只要我们用心感受，就会发现，温暖一直都在。
    给这篇作文打分。
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
