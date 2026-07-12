import csv
import os
import random

from nanodeploy import LLM, SamplingParams
from nanodeploy.engine.sequence import Sequence
from transformers import AutoTokenizer


def main():
    # 模型路径
    model_path = os.path.expanduser(
        "/models/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/ba82a1060073fa0ecdc70d7b1922ec071f60cf3e"
    )
    # 数据集路径
    data_path = "/mnt/nvme1n1/ml_research/majinming/dataset/mixed_dataset_50000total_5long_512Ktotal_0.0weight_20251104_084048.csv"

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 初始化LLM（保留dummy_prefill=True）
    decode = LLM(
        model_path,
        enforce_eager=False,
        attention_dp=32,
        attention_sp=1,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=32,
        ffn_tp=1,
        mode="decode",
        master_address="10.102.207.84:6006",
        ray_address="10.103.5.41:7077",
        dummy_prefill=True,
        max_num_batched_tokens=524288,
        max_num_seqs=128,
        max_model_len=524288,
        gpu_memory_utilization=0.95,
        dummy_weight=True,
        perfect_eplb=True,
    )

    base_sampling_params = SamplingParams(temperature=0.1, ignore_eos=True)

    # 读取数据集并生成序列
    print("preparing dataset begin...")
    sequences = []
    with open(data_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 从数据集中获取长度信息
            prompt_len = int(row["prompt_len"])
            output_len = int(row["output_len"])

            # 生成随机prompt token_ids（长度匹配数据集的prompt_len）
            vocab_size = (
                tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 100000
            )
            prompt_token_ids = [0 for _ in range(min(prompt_len, 524288 - 16384))]

            # 为每个样本设置匹配output_len的max_tokens
            sample_sampling_params = SamplingParams(
                temperature=base_sampling_params.temperature,
                max_tokens=output_len,
                ignore_eos=base_sampling_params.ignore_eos,
            )

            # 创建序列对象
            seq = Sequence(prompt_token_ids, sampling_params=sample_sampling_params)
            sequences.append(seq)
    print("preparing dataset done...")

    # 提交请求并生成
    decode.add_request(sequences)
    decode.generate()

    # # 输出结果
    # for i, seq in enumerate(sequences):
    #     token_ids = seq.completion_token_ids
    #     output = {"text": tokenizer.decode(token_ids), "token_ids": token_ids}
    #     print(
    #         f"Sample {i+1} (type: {reader.fieldnames['type']}) Completion Length: {len(token_ids)}"
    #     )
    #     print(f"Completion: {output['text']!r}\n")


if __name__ == "__main__":
    main()
