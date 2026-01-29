import argparse
import os
import time

from nanodeploy.config import Config
from nanodeploy.engine.sequence import Sequence
from nanodeploy.llm import LLM
from nanodeploy.logging import get_logger
from nanodeploy.sampling_params import SamplingParams
from transformers import AutoTokenizer

logger = get_logger("nanodeploy")


def main():
    parser = argparse.ArgumentParser(
        description="Run a single engine node for etcd mesh."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["prefill", "decode"],
        help="Engine role",
    )
    parser.add_argument(
        "--etcd-address",
        type=str,
        default="10.102.97.179:2379",
        help="Etcd server address (host:port)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/models/qwen3-235B-Instruct-2507-FP8",
        help="Model path",
    )
    parser.add_argument(
        "--master-address",
        type=str,
        default="10.102.97.179:6006",
        help="Ray master address",
    )
    parser.add_argument(
        "--ray-address", type=str, default="10.102.97.179:7078", help="Ray head address"
    )

    args = parser.parse_args()

    config = Config(
        model=args.model,
        enforce_eager=True,
        attention_dp=1 if args.mode == "decode" else 8,
        attention_sp=8 if args.mode == "decode" else 1,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
        mode=args.mode,
        loop_count=16 if args.mode == "decode" else 1,
        master_address=args.master_address,
        ray_address=args.ray_address,
        gpu_memory_utilization=0.5,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        dummy_weight=False,
        log_level="INFO",
        etcd_address=args.etcd_address,
        cluster_id="default",
    )

    logger.info(f"Starting {args.mode.upper()} engine locally...")
    engine = LLM(config)

    logger.info("Waiting for mesh...")
    engine.wait_for_mesh(expected_peers=1)

    if args.mode == "prefill":
        logger.info("Prefill engine ready. Waiting for requests or running test...")
        # Simple test loop
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        sampling_params = SamplingParams(temperature=0.1, max_tokens=128)
        prompt = "Hello, world!"
        seq = Sequence(tokenizer.encode(prompt), sampling_params=sampling_params)
        engine.add_request(seq)

        while not engine.is_finished():
            step_res = engine.step()
            # step() returns (dp_seqs, outputs, num_tokens, bs, sch_ms, post_sch_ms)
            outputs = step_res[1]
            for seq in outputs:
                print(
                    f"Finished Seq {seq.seq_id}: {tokenizer.decode(seq.completion_token_ids)}"
                )

    else:
        logger.info("Decode engine ready. Idle loop...")
        while True:
            engine.step()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
