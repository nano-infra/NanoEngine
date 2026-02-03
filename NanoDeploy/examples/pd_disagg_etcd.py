import os
import time

import ray
from nanodeploy.config import Config
from nanodeploy.engine.sequence import Sequence
from nanodeploy.llm import LLM
from nanodeploy.logging import get_logger
from nanodeploy.sampling_params import SamplingParams
from transformers import AutoTokenizer

logger = get_logger("nanodeploy")


def main():
    # Use a real model path or a dummy one if needed.
    # For this example, we'll assume the same path as in the original pd_disagg.py
    path = os.path.expanduser("/models/qwen3-235B-Instruct-2507-FP8")
    tokenizer = AutoTokenizer.from_pretrained(path)

    decode_config = Config(
        model=path,
        enforce_eager=False,
        attention_dp=1,
        attention_sp=8,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
        mode="decode",
        loop_count=16,
        master_address="10.102.97.179:6006",
        ray_address="10.102.97.179:7078",
        dummy_prefill=False,
        max_num_seqs=128,
        gpu_memory_utilization=0.5,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        dummy_weight=False,
        log_level="INFO",
    )

    prefill_config = Config(
        model=path,
        enforce_eager=True,
        loop_count=1,
        attention_dp=8,
        attention_sp=1,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
        mode="prefill",
        master_address="10.102.97.183:6006",
        ray_address="10.102.97.179:7078",
        gpu_memory_utilization=0.5,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        dummy_weight=False,
        log_level="INFO",
    )

    # We start decode node first
    logger.info("Starting DECODE node...")
    decode = LLM.as_remote(decode_config)

    # Start prefill node
    logger.info("Starting PREFILL node...")
    prefill = LLM.as_remote(prefill_config)

    # ==========================================================
    # AUTOMATED MESH BARRIER
    # ==========================================================
    logger.info("Waiting for automated P2P mesh to be established via etcd...")

    # Wait for the mesh to be ready.
    # expected_peers=1 means we expect this node to connect to 1 other engine.
    ray.get(decode.wait_for_mesh.remote(expected_peers=1))
    ray.get(prefill.wait_for_mesh.remote(expected_peers=1))

    logger.info("Mesh established! Proceeding with inference.")

    sampling_params = SamplingParams(temperature=0.1, max_tokens=128, ignore_eos=False)
    prompts = ["Tell me a short story about an AI that discovered etcd."]

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

    # Run Prefill
    ray.get(prefill.add_request.remote(seqs))
    migrated_seqs = ray.get(prefill.generate.remote())

    if not migrated_seqs:
        logger.error("No sequences migrated from prefill.")
        return

    # Run Decode
    ray.get(decode.add_request.remote(migrated_seqs))
    finished_seqs = ray.get(decode.generate.remote())

    # Finish
    ray.get(prefill.free_to_be_migrated.remote(migrated_seqs))

    for seq in finished_seqs:
        print(
            f"Seq ID: {seq.seq_id} | Output: {tokenizer.decode(seq.completion_token_ids)!r}"
        )


if __name__ == "__main__":
    main()
