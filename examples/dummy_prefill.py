import os

from nanodeploy import LLM, SamplingParams
from nanodeploy.engine.sequence import Sequence
import numpy as np
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/models/qwen3-235B-Instruct-2507-FP8")

    decode = LLM(
        path,
        enforce_eager=False,
        attention_dp=2,
        attention_sp=8,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=16,
        ffn_tp=1,
        mode="decode",
        master_address='10.102.98.166:26444',
        ray_address=   '10.102.98.166:6444',
        dummy_prefill=True,
        dummy_weight=True,
        perfect_eplb=True,
        max_num_seqs=128,
        max_model_len=524288,
        max_num_batched_tokens=524288,
        loop_count=16,
    )

    sampling_params = SamplingParams(temperature=0.1, max_tokens=256, ignore_eos=True)

    long_seqs = [
        Sequence(
            np.random.randint(0, 10001, size=400000).tolist(),
            # [0] * 400000,
            sampling_params=sampling_params,
        )
    ]
    long_seqs = []

    short_seqs = [
        Sequence(
            np.random.randint(0, 10001, size=2048).tolist(),
            # [0] * 2048,
            sampling_params=sampling_params,
        )
        for _ in range(1024)
    ]

    seqs = long_seqs + short_seqs

    decode.add_request(seqs)
    decode.generate()


if __name__ == "__main__":
    main()
