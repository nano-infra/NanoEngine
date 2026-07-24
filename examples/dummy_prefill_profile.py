"""
Custom profiling script for NanoDeploy with DP32/DP4SP8 configurations.

Supports:
- DP32: attention_dp=32, attention_sp=1
- DP4SP8: attention_dp=4, attention_sp=8
- enable_non_uniform_split=True
- sp_master_selector="LeastBatch"
- Custom sp_seq_lens input via JSON file
- Profiling on specific loop iterations (default: loops 5-6)
"""

import argparse
import json
import os

import numpy as np

from nanodeploy import LLM, SamplingParams
from nanodeploy.engine.sequence import Sequence


# Default sp_seq_lens data (provided by user)
DEFAULT_SP_SEQ_LENS = [
    [[313929, 12195, 12471, 11963, 12231]],
    [[12818, 12264, 12014, 12502, 312842]],
    [[142548, 636997, 12297, 316205, 12241]],
    [[12802, 11869, 12264, 12379, 11909]],
    [[353357, 12517, 11931, 11934, 12513]],
    [[306959, 12262, 12587, 12099, 12209]],
    [[12754, 11853, 12647, 12263, 12410]],
    [[972921, 11998, 12722, 12439, 12202]],
    [[239629, 11762, 12109, 310265, 12446]],
    [[12693, 12321, 12363, 12198, 12653]],
    [[592259, 12490, 12185, 12163, 11966]],
    [[887245, 12566, 12649, 12494, 11634]],
    [[12629, 12239, 12185, 12232, 920349]],
    [[12625, 11998, 12685, 12437, 12394]],
    [[12141, 12618, 12234, 12701, 12154]],
    [[972885, 12372, 12550, 12722, 12083]],
    [[11749, 408810, 11947, 12168, 12432]],
    [[12693, 312931, 12257, 12214, 12154]],
    [[12296, 11982, 12331, 12152]],
    [[12587, 12439, 12464, 11950, 12169]],
    [[12587, 12394, 737664, 12378, 12469]],
    [[12280, 12706, 11966, 12669, 12432]],
    [[12264, 12194, 11698, 9872]],
    [[333174, 12561, 11933, 12078]],
    [[533172, 12693, 11917, 10761]],
    [[142220, 11931, 12309, 12513]],
    [[12103, 12326, 11877, 12152]],
    [[12248, 12194, 12198, 12470]],
    [[12119, 11995, 11682, 12152]],
    [[12597, 12007, 12674, 12674]],
    [[12590, 12754, 12125, 11861]],
    [[386118, 12646, 12560, 12645]],
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Profiling script with DP32/DP4SP8 configurations"
    )
    
    # Configuration preset
    parser.add_argument(
        "--config",
        type=str,
        default="dp32leastbatch",
        choices=["dp32", "dp32leastbatch", "dp32leastcache", "dp4sp8"],
        help="Parallelism configuration: dp32/dp32leastbatch, dp32leastcache or dp4sp8"
    )
    
    # Model and deployment settings
    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3",
        help="Path to the model"
    )
    parser.add_argument(
        "--master-address",
        type=str,
        default="10.102.97.179:26444",
        help="Master address for distributed setup"
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default="10.102.97.179:6444",
        help="Ray cluster address"
    )
    
    # Custom seq_lens input
    parser.add_argument(
        "--sp-seq-lens-file",
        type=str,
        default=None,
        help="Path to JSON file containing sp_seq_lens list"
    )
    
    # Profiler settings
    parser.add_argument(
        "--profiler-dir",
        type=str,
        default="/mnt/nvme1n1/ml_research/linbinbin1/profiler_res",
        help="Directory to save profiler results"
    )
    parser.add_argument(
        "--profiler-start-step",
        type=int,
        default=5,
        help="Start profiling at this step (0-indexed, default: 5)"
    )
    parser.add_argument(
        "--profiling-step",
        type=int,
        default=2,
        help="Number of steps to profile (default: 2, profiles step 5-6)"
    )
    
    # Generation settings
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Output tokens length (default: 256)"
    )
    parser.add_argument(
        "--loop-count",
        type=int,
        default=16,
        help="Number of decode iterations per step (default: 48)"
    )
    
    # Other settings
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=128,
        help="Maximum number of sequences per batch"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1_000_000,
        help="Maximum model sequence length"
    )
    
    return parser.parse_args()


def get_config_params(config_name: str) -> dict:
    """Get parallelism parameters for the given configuration."""
    if config_name in ["dp32", "dp32leastbatch", "dp32leastcache"]:
        return {
            "attention_dp": 32,
            "attention_sp": 1,
            "attention_tp": 1,
            "ffn_dp": 1,
            "ffn_ep": 32,
            "ffn_tp": 1,
        }
    elif config_name == "dp4sp8":
        return {
            "attention_dp": 4,
            "attention_sp": 8,
            "attention_tp": 1,
            "ffn_dp": 1,
            "ffn_ep": 32,
            "ffn_tp": 1,
        }
    else:
        raise ValueError(f"Unknown config: {config_name}")


def load_sp_seq_lens(file_path: str | None) -> list:
    """Load sp_seq_lens from JSON file or return default."""
    if file_path is None:
        print("Using default embedded sp_seq_lens data")
        return DEFAULT_SP_SEQ_LENS
    
    print(f"Loading sp_seq_lens from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'sp_seq_lens' in data:
        return data['sp_seq_lens']
    return data


def create_sequences_from_sp_seq_lens(sp_seq_lens: list, sampling_params: SamplingParams) -> list[Sequence]:
    """
    Create sequences based on sp_seq_lens structure.
    
    Each entry in sp_seq_lens corresponds to a different request/batch configuration.
    We create sequences with lengths derived from the seq_lens.
    """
    sequences = []
    
    for loop_idx, loop_data in enumerate(sp_seq_lens):
        # loop_data is a list of DP groups, each containing SP ranks' seq lens
        for dp_idx, dp_group in enumerate(loop_data):
            for seq_len in dp_group:
                # Create a sequence with the specified length
                token_ids = np.random.randint(0, 10001, size=seq_len).tolist()
                seq = Sequence(token_ids, sampling_params=sampling_params)
                sequences.append(seq)
    
    return sequences


def main():
    args = parse_args()
    
    # Update profiler dir to include config name
    args.profiler_dir = os.path.join(args.profiler_dir, args.config)
    
    # Get parallelism configuration
    config_params = get_config_params(args.config)
    
    print(f"\n{'='*60}")
    print(f"Configuration: {args.config}")
    print(f"  attention_dp={config_params['attention_dp']}")
    print(f"  attention_sp={config_params['attention_sp']}")
    print(f"  ffn_ep={config_params['ffn_ep']}")
    print(f"  enable_non_uniform_split=True")
    
    # Determine selector
    selector = "LeastBatch"
    if args.config == "dp32leastcache":
        selector = "LeastCache"

    print(f"  sp_master_selector={selector}")
    print("  scheduler_arch=legacy_global")
    print(f"{'='*60}\n")
    
    # Initialize the LLM engine
    print(f"\nInitializing LLM engine...")
    print(f"  Profiler enabled: True")
    print(f"  Profiler start step: {args.profiler_start_step}")
    print(f"  Profiling steps: {args.profiling_step}")
    print(f"  Profiler dir: {args.profiler_dir}")
    
    decode = LLM(
        args.model_path,
        enforce_eager=False,
        **config_params,
        mode="decode",
        master_address=args.master_address,
        ray_address=args.ray_address,
        dummy_prefill=True,
        dummy_weight=True,
        perfect_eplb=True,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        loop_count=args.loop_count,
        max_num_send_seqs=48,
        max_num_recv_seqs=48,
        kvcache_block_size=64,
        # Profiler settings
        enable_profiler=True,
        profiler_start_step=args.profiler_start_step,
        profiling_step=args.profiling_step,
        profiler_dir=args.profiler_dir,
        # Non-uniform split and LeastBatch selector
        enable_non_uniform_split=True,
        sp_master_selector=selector,
        scheduler_arch="legacy_global",
        routing_strategy=selector,
    )
    
    print(f"\nEngine initialized successfully!")
    print(f"Starting profiling with {args.max_tokens} output tokens...")
    
    # Create sampling params
    sampling_params = SamplingParams(
        temperature=0.1, 
        max_tokens=args.max_tokens, 
        ignore_eos=True
    )
    
    sp_seq_lens = load_sp_seq_lens(args.sp_seq_lens_file)
    print(f"Loaded {len(sp_seq_lens)} loop configurations")

    total_seqs = 0
    for loop_data in sp_seq_lens:
        for dp_group in loop_data:
            total_seqs += len(dp_group)

    print(f"Total sequences from sp_seq_lens: {total_seqs}")

    sequences = []
    for dp_group in sp_seq_lens:
        for sp_rank_seqs in dp_group:
            for seq_len in sp_rank_seqs:
                token_ids = np.random.randint(0, 10001, size=seq_len).tolist()
                seq = Sequence(token_ids, sampling_params=sampling_params)
                sequences.append(seq)

    print(f"Created {len(sequences)} sequences from sp_seq_lens for profiling")
    decode.add_request(sequences)
    
    decode.generate()
    
    print(f"\n{'='*60}")
    print(f"Profiling completed!")
    print(f"Results saved to: {args.profiler_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
