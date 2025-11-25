import os

from nanodeploy import LLM, SamplingParams
from nanodeploy.engine.sequence import Sequence
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser(
        # "/models/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/ba82a1060073fa0ecdc70d7b1922ec071f60cf3e"
        "/models/model-deepseek-v3-no-symlink"
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
        mode="decode",
        master_address="10.102.206.55:6006",
        ray_address="10.102.206.55:7777",
        dummy_prefill=True,
        dummy_weight=True,
        perfect_eplb=True,
        max_num_seqs=64,
        max_model_len=524288,
        max_num_batched_tokens=524288,
        loop_count=48,
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(temperature=0.1, max_tokens=256, ignore_eos=True)

    long_seqs = [
        Sequence(
            [0] * 400000,
            sampling_params=sampling_params,
        )
    ]
    long_seqs = []

    short_seqs = [
        Sequence(
            [0] * 1024,
            sampling_params=sampling_params,
        )
        for _ in range(512)
    ]

    seqs = long_seqs + short_seqs

    decode.add_request(seqs)
    decode.generate()


if __name__ == "__main__":
    main()
