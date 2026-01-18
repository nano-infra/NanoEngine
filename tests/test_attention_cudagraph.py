"""
性能测试：单卡 Attention with CUDA Graph

这个测试用例用于测试单卡（非SP）Attention在使用CUDA Graph时的性能。
支持两种Attention类型：
1. GQA (Grouped Query Attention) - 使用FlashAttention
2. MLA (Multi-head Latent Attention) - 使用FlashMLA

测试包括：
1. Attention计算（FlashAttention或FlashMLA）
2. CUDA Graph的capture和replay

使用方法：
    # 测试GQA
    python tests/test_attention_cudagraph.py --attention_type GQA --num_kv_heads 8
    
    # 测试MLA（要求num_kv_heads=1, head_dim=576, v_head_dim=512）
    python tests/test_attention_cudagraph.py \
        --attention_type MLA \
        --num_kv_heads 1 \
        --head_dim 576 \
        --v_head_dim 512
"""

import os
import csv
from datetime import datetime
from typing import Optional, Literal

import torch
from flash_attn_interface import flash_attn_with_kvcache
from triton.testing import do_bench

try:
    import flash_mla
except ImportError:
    flash_mla = None


def create_test_data(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    seq_len: int,
    attention_type: Literal["GQA", "MLA"] = "GQA",
    v_head_dim: Optional[int] = None,
    page_size: int = 16,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """创建测试数据"""
    # Q tensor
    q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)

    # K cache
    # For MLA: k_cache stores K, so it should use head_dim (576), not v_head_dim (512)
    k_cache = torch.randn(
        batch_size * seq_len // page_size,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    
    # V cache (only for GQA, MLA doesn't need v_cache storage)
    if attention_type == "GQA":
        v_cache = torch.randn(
            batch_size * seq_len // page_size,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
    else:
        # MLA doesn't use v_cache, but we'll create a dummy tensor for compatibility
        v_cache = torch.tensor([], dtype=dtype, device=device)

    # Block tables
    num_pages = seq_len // page_size
    block_tables = torch.arange(
        batch_size * num_pages, device=device, dtype=torch.int32
    ).view(batch_size, num_pages)

    # Context lengths
    context_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "context_lens": context_lens,
        "block_tables": block_tables,
    }


def attention_forward_gqa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    num_heads: int,
    head_dim: int,
    scale: float,
) -> torch.Tensor:
    """
    Attention Forward Pass (GQA with FlashAttention)
    
    Args:
        q: Query tensor [batch_size, num_heads, head_dim]
        k_cache: Key cache tensor
        v_cache: Value cache tensor
        context_lens: Context lengths for attention computation
        block_tables: Block tables for paged attention
        num_heads: Number of attention heads
        head_dim: Head dimension
        scale: Attention scale
        
    Returns:
        Output tensor [batch_size, num_heads, head_dim]
    """
    # Attention computation
    o, _ = flash_attn_with_kvcache(
        q.unsqueeze(1),
        k_cache,
        v_cache,
        cache_seqlens=context_lens,
        page_table=block_tables,
        softmax_scale=scale,
        causal=True,
        return_softmax_lse=True,
    )[:2]
    
    o = o.squeeze(1)
    return o


def attention_forward_mla(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    num_heads: int,
    head_dim: int,
    v_head_dim: int,
    scale: float,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    debug: bool = False,
) -> torch.Tensor:
    """
    Attention Forward Pass (MLA with FlashMLA)
    
    Args:
        q: Query tensor [batch_size, num_heads, head_dim]
        k_cache: Key cache tensor (MLA only needs k_cache, not v_cache)
        context_lens: Context lengths for attention computation
        block_tables: Block tables for paged attention
        num_heads: Number of attention heads
        head_dim: Head dimension (for Q and K)
        v_head_dim: V head dimension (for output)
        scale: Attention scale
        tile_scheduler_metadata: Precomputed MLA metadata (computed once during initialization)
        num_splits: Precomputed MLA num_splits (computed once during initialization)
        debug: Enable debug logging
        
    Returns:
        Output tensor [batch_size, num_heads, v_head_dim]
    """
    # tile_scheduler_metadata and num_splits are precomputed during initialization
    # and passed as parameters to avoid recalculating on each forward pass

    # Ensure q.unsqueeze(1) has the correct shape before calling FlashMLA
    q_for_attn = q.unsqueeze(1)
    
    # Debug logs - match format with FlashMLAImpl
    if debug:
        print(f"[TEST] Before flash_mla_with_kvcache:")
        print(f"  q.shape: {q.shape}, q_for_attn.shape: {q_for_attn.shape}")
        print(f"  k_cache.shape: {k_cache.shape}")
        print(f"  k_cache last dimension: {k_cache.shape[-1]}")
        print(f"  block_tables.shape: {block_tables.shape}")
        print(f"  context_lens.shape: {context_lens.shape}")
        print(f"  context_lens: {context_lens}")
        print(f"  v_head_dim parameter: {v_head_dim} (type: {type(v_head_dim)})")
        print(f"  head_dim parameter: {head_dim} (type: {type(head_dim)})")
        print(f"  num_heads: {num_heads}, num_kv_heads: 1")
        print(f"  scale: {scale}, causal: True")
        print(f"  tile_scheduler_metadata type: {type(tile_scheduler_metadata)}")
        print(f"  num_splits.shape: {num_splits.shape if hasattr(num_splits, 'shape') else type(num_splits)}")
    
    # For MLA: k_cache uses head_dim (576) for K, but v_head_dim (512) is passed to flash_mla_with_kvcache
    assert k_cache.shape[-1] == head_dim, (
        f"k_cache last dimension ({k_cache.shape[-1]}) must equal head_dim ({head_dim}) for K storage"
    )
    
    o, lse = flash_mla.flash_mla_with_kvcache(
        q_for_attn,
        k_cache,
        block_tables,
        context_lens,
        v_head_dim,
        tile_scheduler_metadata,
        num_splits,
        scale,
        causal=True,
    )
    
    if debug:
        print(f"[TEST] After flash_mla_with_kvcache:")
        print(f"  o.shape (before squeeze): {o.shape}")
        print(f"  lse.shape: {lse.shape}")

    o = o.squeeze(1)
    if debug:
        print(f"  o.shape (after squeeze): {o.shape}")
    
    return o


def benchmark_attention_with_cudagraph(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    attention_type: Literal["GQA", "MLA"] = "GQA",
    v_head_dim: Optional[int] = None,
    batch_size: int = 1,
    use_cudagraph: bool = True,
    num_warmup: int = 100,
    num_iterations: int = 200,
    debug: bool = False,
    enable_profiler: bool = False,
    profiler_iterations: int = 50,
):
    """
    使用CUDA Graph对单卡Attention进行性能测试
    
    Args:
        seq_len: Sequence length
        num_heads: Number of query heads
        head_dim: Head dimension (for Q and K)
        num_kv_heads: Number of key/value heads (must be 1 for MLA)
        attention_type: Attention type ("GQA" or "MLA")
        v_head_dim: V head dimension (for MLA output, if None, uses head_dim)
        batch_size: Batch size
        use_cudagraph: Whether to use CUDA Graph
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        debug: Enable debug logging for tensor shapes
        enable_profiler: Whether to run profiler after benchmark
        profiler_iterations: Number of iterations to profile (default: 50)
    """
    # Validate attention type
    if attention_type == "MLA":
        if num_kv_heads != 1:
            raise ValueError("MLA requires num_kv_heads == 1")
        if flash_mla is None:
            raise ImportError("flash_mla is required for MLA attention. Please install it.")
        if head_dim != 576:
            raise ValueError(
                f"MLA requires head_dim (kv_lora_rank + qk_rope_head_dim) == 576, but got {head_dim}. "
                f"Please set --head_dim 576 when using --attention_type MLA"
            )
    
    device = "cuda"
    dtype = torch.bfloat16
    
    # MLA mode requires block_size=64, GQA can use page_size=16
    if attention_type == "MLA":
        page_size = 64  # MLA mode only supports block_size=64
        # Also need to ensure seq_len is divisible by page_size
        if seq_len % page_size != 0:
            seq_len = ((seq_len + page_size - 1) // page_size) * page_size
            print(f"Warning: seq_len adjusted to {seq_len} to be divisible by page_size={page_size}")
    else:
        page_size = 16  # GQA can use smaller page size
    
    scale = 1.0 / (head_dim**0.5)
    
    # For MLA, v_head_dim is kv_lora_rank = 512, not 576!
    if v_head_dim is None:
        if attention_type == "MLA":
            v_head_dim = 512
        else:
            v_head_dim = head_dim
    
    # Validate v_head_dim for MLA
    if attention_type == "MLA":
        if v_head_dim != 512:
            raise ValueError(
                f"MLA requires v_head_dim (kv_lora_rank) == 512, but got {v_head_dim}. "
                f"Note: head_dim=576 is for Q/K (512+64), but v_head_dim=512 is for output V. "
                f"Please set --v_head_dim 512 when using --attention_type MLA"
            )
        print(f"Validation passed: head_dim={head_dim} (for Q/K), v_head_dim={v_head_dim} (for output V)")

    print(f"\n{'='*80}")
    print(f"Single-GPU Attention Performance Test with CUDA Graph")
    print(f"{'='*80}")
    print(f"Attention Type: {attention_type}")
    print(f"Batch Size: {batch_size}")
    print(f"Sequence Length: {seq_len}")
    print(f"Num Heads: {num_heads}, Num KV Heads: {num_kv_heads}")
    print(f"Head Dim: {head_dim}")
    if attention_type == "MLA":
        print(f"V Head Dim: {v_head_dim}")
    print(f"Use CUDA Graph: {use_cudagraph}")
    print(f"{'='*80}\n")

    # 创建测试数据
    test_data = create_test_data(
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        seq_len=seq_len,
        attention_type=attention_type,
        v_head_dim=v_head_dim,
        page_size=page_size,
        device=device,
        dtype=dtype,
    )

    # Precompute MLA metadata if needed (only once during initialization)
    tile_scheduler_metadata = None
    num_splits = None
    if attention_type == "MLA":
        # For MLA: num_query_heads_per_kv = num_heads // num_kv_heads = num_heads // 1 = num_heads
        num_query_heads_per_kv = num_heads // 1
        if debug:
            print(f"[TEST] Precomputing get_mla_metadata: context_lens.shape={test_data['context_lens'].shape}, "
                  f"context_lens={test_data['context_lens']}, "
                  f"num_query_heads_per_kv={num_query_heads_per_kv}, num_kv_heads=1")
        tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
            test_data["context_lens"],
            num_query_heads_per_kv,
            1,  # num_kv_heads (MLA requires num_kv_heads == 1)
        )
    
    # Debug print before creating forward_fn (to avoid printing during CUDA Graph capture)
    if debug:
        print("[TEST] Debug info (before CUDA Graph capture):")
        print(f"  context_lens: {test_data['context_lens']}")
        print(f"  block_tables.shape: {test_data['block_tables'].shape}")
        if attention_type == "MLA":
            print(f"  tile_scheduler_metadata type: {type(tile_scheduler_metadata)}")
            if hasattr(num_splits, 'shape'):
                print(f"  num_splits.shape: {num_splits.shape}")
    
    # 创建forward函数（注意：在CUDA Graph capture期间，debug打印会被禁用以避免CUDA错误）
    if attention_type == "GQA":
        def forward_fn():
            return attention_forward_gqa(
                q=test_data["q"],
                k_cache=test_data["k_cache"],
                v_cache=test_data["v_cache"],
                context_lens=test_data["context_lens"],
                block_tables=test_data["block_tables"],
                num_heads=num_heads,
                head_dim=head_dim,
                scale=scale,
            )
    else:  # MLA
        def forward_fn():
            return attention_forward_mla(
                q=test_data["q"],
                k_cache=test_data["k_cache"],
                context_lens=test_data["context_lens"],
                block_tables=test_data["block_tables"],
                num_heads=num_heads,
                head_dim=head_dim,
                v_head_dim=v_head_dim,
                scale=scale,
                tile_scheduler_metadata=tile_scheduler_metadata,
                num_splits=num_splits,
                debug=False,  # 在forward_fn中禁用debug，避免CUDA Graph capture期间的tensor打印
            )

    # Warmup
    print("Warming up...")
    for _ in range(num_warmup):
        _ = forward_fn()
    torch.cuda.synchronize()

    # Initialize graph variable for potential profiler use
    graph = None
    
    # Benchmark with or without CUDA Graph
    if use_cudagraph:
        print(f"Benchmarking with CUDA Graph ({num_iterations} iterations)...")
        torch.cuda.synchronize()
        
        # Capture CUDA Graph
        graph = torch.cuda.CUDAGraph()
        graph_pool = None
        
        # Warmup for graph capture
        for _ in range(10):
            _ = forward_fn()
        torch.cuda.synchronize()
        
        # Capture graph
        with torch.cuda.graph(graph, graph_pool):
            output = forward_fn()
        
        if graph_pool is None:
            graph_pool = graph.pool()
        
        # Benchmark with graph replay
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_iterations):
            graph.replay()
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = elapsed_time_ms / num_iterations
        avg_time_us = avg_time_ms * 1000
        
        print(f"Average time with CUDA Graph: {avg_time_us:.2f} us")
    else:
        print(f"Benchmarking without CUDA Graph ({num_iterations} iterations)...")
        elapsed_time = do_bench(forward_fn, warmup=10, rep=num_iterations)
        avg_time_us = elapsed_time * 1e6
        print(f"Average time without CUDA Graph: {avg_time_us:.2f} us")

    # 计算吞吐量和理论性能
    total_seqlen = seq_len * batch_size
    if attention_type == "GQA":
        mem_io = (
            total_seqlen * num_kv_heads * head_dim * 2  # K, V cache
            + batch_size * num_heads * head_dim * 2  # Q
            + batch_size * num_heads * head_dim * 2  # Output
        )
    else:  # MLA
        mem_io = (
            total_seqlen * num_kv_heads * head_dim  # K cache
            + batch_size * num_heads * head_dim * 2  # Q
            + batch_size * num_heads * v_head_dim * 2  # Output
        )
    
    if attention_type == "GQA":
        flops = (
            batch_size
            * total_seqlen
            * num_heads
            * head_dim
            * 2
        )
    else:  # MLA
        flops = (
            batch_size
            * total_seqlen
            * num_heads
            * v_head_dim
            * 2
        )
    
    throughput_gbs = mem_io * 1e-9 / (avg_time_us * 1e-6)
    throughput_tflops = flops * 1e-12 / (avg_time_us * 1e-6)
    
    # 理论性能（假设H100）
    ideal_h100_time_mem = mem_io / 3.35e12 * 1e6  # microseconds
    ideal_h100_time_flop = flops / 989e12 * 1e6  # microseconds
    ideal_h100_time = max(ideal_h100_time_mem, ideal_h100_time_flop)
    
    print(f"\n{'='*80}")
    print(f"Performance Results:")
    print(f"{'='*80}")
    print(f"Average Latency: {avg_time_us:.2f} us")
    print(f"Throughput: {throughput_gbs:.2f} GB/s")
    print(f"Compute: {throughput_tflops:.2f} TFLOPS/s")
    print(f"Arithmetic Intensity: {flops / mem_io:.2f}")
    print(f"Ideal H100 Time: {ideal_h100_time:.2f} us")
    print(f"Efficiency: {ideal_h100_time / avg_time_us * 100:.1f}%")
    print(f"{'='*80}\n")

    # Profiler (if enabled)
    if enable_profiler:
        print(f"Running profiler ({profiler_iterations} iterations)...")
        torch.cuda.synchronize()
        
        # Generate output filename with timestamp and mode
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_str = "eager" if not use_cudagraph else "graph"
        output_dir = f"profiler_traces/single_gpu/{mode_str}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            f"attention_{attention_type.lower()}_seq{seq_len}_{mode_str}_{timestamp}.json"
        )
        
        # Create profiler
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        
        if use_cudagraph:
            # For CUDA Graph, profile the graph replay
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                for iter_idx in range(profiler_iterations):
                    # Mark iteration boundary in profiler trace
                    with torch.profiler.record_function(f"iteration_{iter_idx}"):
                        graph.replay()
                torch.cuda.synchronize()
        else:
            # For non-graph mode, profile the forward function
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                for iter_idx in range(profiler_iterations):
                    # Mark iteration boundary in profiler trace
                    with torch.profiler.record_function(f"iteration_{iter_idx}"):
                        _ = forward_fn()
                torch.cuda.synchronize()
        
        # Export chrome trace
        prof.export_chrome_trace(output_file)
        print(f"Profiler trace saved to: {output_file}")
        
        # Get all kernel statistics
        key_averages = prof.key_averages(group_by_input_shape=True)
        
        # Save kernel statistics to CSV
        csv_file = output_file.replace('.json', '.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow([
                'Name',
                'Self CPU Time (us)',
                'CPU Time (us)',
                'Self CUDA Time (us)',
                'CUDA Time (us)',
                'Avg Self CUDA Time (us)',
                'Self CPU Memory (bytes)',
                'CPU Memory (bytes)',
                'Self CUDA Memory (bytes)',
                'CUDA Memory (bytes)',
                'Input Shapes',
                'Call Count',
            ])
            
            # Write data rows
            for event in key_averages:
                # Extract input shapes if available
                input_shapes = ''
                if hasattr(event, 'input_shapes') and event.input_shapes:
                    input_shapes = str(event.input_shapes)
                
                # Get time values (already in microseconds for torch.profiler)
                self_cpu_time = getattr(event, 'self_cpu_time_total', 0)
                cpu_time = getattr(event, 'cpu_time_total', 0)
                self_cuda_time = getattr(event, 'self_cuda_time_total', 0)
                cuda_time = getattr(event, 'cuda_time_total', 0)
                count = getattr(event, 'count', 1)
                avg_self_cuda_time = self_cuda_time / count if count > 0 else 0
                
                # Get memory values (in bytes)
                self_cpu_mem = getattr(event, 'self_cpu_memory_usage', 0)
                cpu_mem = getattr(event, 'cpu_memory_usage', 0)
                self_cuda_mem = getattr(event, 'self_cuda_memory_usage', 0)
                cuda_mem = getattr(event, 'cuda_memory_usage', 0)
                
                writer.writerow([
                    event.key,  # Kernel/operator name
                    f'{self_cpu_time:.2f}',
                    f'{cpu_time:.2f}',
                    f'{self_cuda_time:.2f}',
                    f'{cuda_time:.2f}',
                    f'{avg_self_cuda_time:.2f}',
                    self_cpu_mem,
                    cpu_mem,
                    self_cuda_mem,
                    cuda_mem,
                    input_shapes,
                    count,
                ])
        
        print(f"Kernel statistics saved to: {csv_file}")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Profiler Summary (top 10 operators):")
        print(f"{'='*80}")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(f"{'='*80}\n")


def main():
    """主函数：运行测试"""
    import argparse

    parser = argparse.ArgumentParser(description="Single-GPU Attention CUDA Graph Performance Test")
    parser.add_argument("--seq_len", type=int, default=None, help="Sequence length (single value)")
    parser.add_argument("--seq_lens", type=str, default=None, help="Comma-separated sequence lengths for batch testing (e.g., '1024,2048,4096,8192')")
    parser.add_argument("--num_heads", type=int, default=64, help="Number of attention heads")
    parser.add_argument("--num_kv_heads", type=int, default=8, help="Number of KV heads (must be 1 for MLA)")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension (for Q and K). For MLA, must be 576 (512+64)")
    parser.add_argument("--v_head_dim", type=int, default=None, help="V head dimension. For MLA, must be 512 (kv_lora_rank), not 576! Default: 512 for MLA, head_dim for GQA")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--attention_type", type=str, default="GQA", choices=["GQA", "MLA"], help="Attention type (GQA or MLA)")
    parser.add_argument("--no_cudagraph", action="store_true", help="Disable CUDA Graph")
    parser.add_argument("--eager", action="store_true", help="Use eager mode (equivalent to --no_cudagraph)")
    parser.add_argument("--iterations", type=int, default=200, help="Number of iterations")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging for tensor shapes")
    parser.add_argument("--enable_profiler", action="store_true", help="Enable profiler to save trace data")
    parser.add_argument("--profiler_iterations", type=int, default=50, help="Number of iterations to profile (default: 50)")
    
    args = parser.parse_args()
    
    # Parse seq_lens: support both --seq_len (single) and --seq_lens (comma-separated)
    if args.seq_lens:
        # Parse comma-separated values
        seq_lens = [int(x.strip()) for x in args.seq_lens.split(',')]
        if args.seq_len is not None:
            print("Warning: Both --seq_len and --seq_lens provided. Using --seq_lens.")
    elif args.seq_len is not None:
        seq_lens = [args.seq_len]
    else:
        # Default to 4096 if neither is provided
        seq_lens = [4096]
    
    # Run benchmarks for each seq_len
    total_runs = len(seq_lens)
    for idx, seq_len in enumerate(seq_lens, 1):
        print(f"\n{'='*80}")
        print(f"Running benchmark {idx}/{total_runs}: seq_len={seq_len}")
        print(f"{'='*80}\n")
        
        benchmark_attention_with_cudagraph(
            seq_len=seq_len,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            num_kv_heads=args.num_kv_heads,
            attention_type=args.attention_type,
            v_head_dim=args.v_head_dim,
            batch_size=args.batch_size,
            use_cudagraph=not (args.no_cudagraph or args.eager),
            num_iterations=args.iterations,
            debug=args.debug,
            enable_profiler=args.enable_profiler,
            profiler_iterations=args.profiler_iterations,
        )
        
        if idx < total_runs:
            print(f"\nCompleted {idx}/{total_runs} runs. Continuing to next seq_len...\n")
    
    print(f"\n{'='*80}")
    print(f"All benchmarks completed! Total runs: {total_runs}")
    print(f"Sequence lengths tested: {seq_lens}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
