"""Experimental component boundaries; these do not change the serving runtime."""
from __future__ import annotations

import torch
import torch.distributed as dist


@torch.inference_mode()
def measure_kda(worker):
    from types import SimpleNamespace
    from dlengine.runtime.context.distributed import set_dist_context
    from dlengine.runtime.context.batch import set_batch_context
    from dlengine.runtime.layers import init_backend
    from dlengine.runtime.layers.backends.delta_net.kda import FlashInferKda
    from bench.k3_layer_performance.benchmark_kda_full_breakdown import median_ms

    rows = []
    config = SimpleNamespace(model_type="kimi_k3", hidden_size=7168,
        v_head_dim=128, rms_norm_eps=1e-6,
        linear_attn_config=dict(num_heads=96, head_dim=128,
            value_head_dim=128, short_conv_kernel_size=4, gate_lower_bound=-5.0))
    for tp in sorted(worker.groups):
        from torch.distributed.device_mesh import DeviceMesh
        ctx = set_dist_context(rank=worker.rank, world_size=worker.world, attention_dp=worker.world // tp,
                         attention_tp=tp, ffn_ep=worker.world, initialize_meshes=False)
        ctx.attn_device_mesh = DeviceMesh.from_group(worker.groups[tp], "cuda", mesh_dim_names=("attn_tp",))
        if worker.rank == 0:
            print(f"KDA TP{tp}: constructing production layer", flush=True)
        init_backend()
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            layer = FlashInferKda(0, 0, config).eval()
            for p in layer.parameters():
                p.zero_()
            layer.process_weights_after_loading()
        torch.set_default_dtype(old_dtype)
        heads = 96 // tp
        for prefill, batch, chunk in [(True, 1, c) for c in getattr(worker, "kda_chunks", (1024, 8192, 16384))] + [(False, b, 1) for b in getattr(worker, "kda_batches", (1, 8, 32, 128, 256))]:
            n = batch * chunk
            hidden = torch.zeros(n, 7168, device="cuda", dtype=torch.bfloat16)
            recurrent = torch.zeros(1, batch, heads, 128, 128, device="cuda", dtype=torch.bfloat16)
            conv = torch.zeros(1, batch, 3 * heads * 128, 4, device="cuda", dtype=torch.bfloat16)
            slots = torch.arange(batch, device="cuda", dtype=torch.int32)
            cu_q = torch.arange(0, n + 1, chunk, device="cuda", dtype=torch.int32)
            cu_k = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * 1048576
            set_batch_context(is_prefill=prefill, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=chunk, max_seqlen_k=1048576,
                block_tables=torch.zeros(batch, 1, device="cuda", dtype=torch.int32),
                gdn_conv_states=conv, gdn_recurrent_states=recurrent, gdn_state_slots=slots)

            def restore():
                recurrent.zero_()
                conv.zero_()

            dist.barrier()
            ms = median_ms(lambda: layer(hidden), restore, 3, 15)
            restore()
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            result = layer(hidden)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - baseline
            assert result.shape == hidden.shape and torch.isfinite(result).all().item()
            graph_ms = None
            if not prefill:
                # Warmed production forward captured to remove Python launch gaps.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        layer(hidden)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with torch.cuda.graph(graph, stream=stream):
                    graph_output = layer(hidden)
                dist.barrier()
                graph_ms = worker.time(graph.replay, warmup=3, repeats=30)
                torch.cuda.synchronize()
                assert torch.isfinite(graph_output).all().item()
                del graph, graph_output
            rows.append(dict(rank=worker.rank, tp=tp, dp=worker.world // tp,
                phase="prefill" if prefill else "decode", batch_per_dp=batch,
                chunk=chunk, ms=ms, graph_ms=graph_ms, dynamic_peak_bytes=peak,
                state_bytes=(recurrent.numel() + conv.numel()) * 2))
            if worker.rank == 0:
                print("KDA", rows[-1], flush=True)
            del result, hidden, recurrent, conv
        del layer
        torch.cuda.empty_cache()
    return rows


@torch.inference_mode()
def measure_cp(worker):
    from flash_attn.cute import flash_attn_varlen_func
    from bench.k3_partition.cp_merge import ContextMerger

    rows = []
    for tp, cp in ((16, 1), (8, 2), (4, 4), (2, 8), (1, 16), (4, 1), (8, 1), (4, 2), (2, 4)):
        group = worker.groups.get(cp) if cp > 1 else None
        heads = 96 // tp

        def make(qrows, length):
            torch.manual_seed(1234 + worker.rank // cp)
            q = torch.randn(qrows, heads, 192, device="cuda", dtype=torch.bfloat16)
            torch.manual_seed(5678 + worker.rank)
            k = torch.randn(length // cp, heads, 192, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(length // cp, heads, 128, device="cuda", dtype=torch.bfloat16)
            cq = torch.tensor([0, qrows], device="cuda", dtype=torch.int32)
            ck = torch.tensor([0, length // cp], device="cuda", dtype=torch.int32)

            def local():
                return flash_attn_varlen_func(q, k, v, cu_seqlens_q=cq,
                    cu_seqlens_k=ck, max_seqlen_q=qrows, max_seqlen_k=length // cp,
                    softmax_scale=192 ** -0.5, causal=False, return_lse=True)

            merger = ContextMerger(qrows, heads, 128, group) if cp > 1 else None

            def complete():
                out, lse = local()
                return merger(out, lse) if cp > 1 else out

            return q, k, v, local, complete

        q, k, v, local, complete = make(32, 128 * cp)
        result = complete()
        if cp > 1:
            all_k, all_v = [torch.empty_like(k) for _ in range(cp)], [torch.empty_like(v) for _ in range(cp)]
            dist.all_gather(all_k, k, group=group)
            dist.all_gather(all_v, v, group=group)
            kref, vref = torch.cat(all_k), torch.cat(all_v)
        else:
            kref, vref = k, v
        score = torch.einsum("qhd,khd->hqk", q.float(), kref.float()) * 192 ** -0.5
        ref = torch.einsum("hqk,khd->qhd", score.softmax(-1), vref.float())
        error = (result.float() - ref).abs().max().item()
        torch.testing.assert_close(result.float(), ref, atol=0.006, rtol=0.03)
        del q, k, v, local, complete, result, kref, vref, score, ref
        if cp > 1:
            del all_k, all_v
        for length in (32768, 131072, 1048576):
            for qrows in (1, 128, 1024, 8192):
                q, k, v, local, complete = make(qrows, length)
                local()
                dist.barrier()
                local_ms = worker.time(local, warmup=3, repeats=10)
                dist.barrier()
                full_ms = worker.time(complete, warmup=3, repeats=10)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        complete()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with torch.cuda.graph(graph, stream=stream):
                    graph_output = complete()
                dist.barrier()
                graph_ms = worker.time(graph.replay, warmup=3, repeats=20)
                del graph, graph_output
                rows.append(dict(rank=worker.rank, tp=tp, cp=cp, dp=16 // (tp * cp),
                    queries=qrows, cached_tokens=length, local_attention_ms=local_ms,
                    attention_with_merge_ms=full_ms, graph_ms=graph_ms, correctness_max_abs=error,
                    expanded_kv_bytes=(k.numel() + v.numel()) * 2,
                    local_attention_flops=2 * heads * (192 + 128) * qrows * length // cp))
                if worker.rank == 0:
                    print("CP", rows[-1], flush=True)
                del q, k, v, local, complete
            torch.cuda.empty_cache()
    return rows


@torch.inference_mode()
def measure_moe(worker):
    from dlengine.runtime.layers.backends.experts.mega_moe import MegaMoEExperts
    from dlengine.runtime.models.quant_config import QuantizationConfig
    from dlengine.runtime.runner.runner_config import set_runner_config

    import torch.distributed._symmetric_memory as symm_mem
    symm_mem.set_backend("NCCL")
    rows = []
    for ep in (16, 8, 4):
        group = worker.groups[ep]
        rank = dist.get_rank(group)
        set_runner_config(mega_moe_max_tokens_per_rank=2048)
        quant = QuantizationConfig(format="mxfp4-pack-quantized",
            config_groups={"g": {"weights": {"group_size": 32}}})
        with torch.device("cuda"):
            experts = MegaMoEExperts(hidden_size=3584, intermediate_size=3072,
                num_experts=896, top_k=16, ep_size=ep, tp_size=1,
                ep_group=group, quantization_config=quant)
        experts.gate_up_proj.zero_()
        experts.down_proj.zero_()
        experts.gate_up_scale.fill_(127)
        experts.down_scale.fill_(127)
        experts.prepare_mega_weights()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free_before = torch.cuda.mem_get_info()[0]
        buffer = experts._get_buffer()
        torch.cuda.synchronize()
        workspace = free_before - torch.cuda.mem_get_info()[0]
        if worker.rank == 0:
            print(f"MegaMoE EP{ep} buffer ready: {workspace / 2**30:.3f} GiB", flush=True)
        for tokens in (1, 8, 128, 512, 2048):
            for routing in ("balanced", "hot_rank"):
                x = torch.zeros(tokens, 3584, device="cuda", dtype=torch.bfloat16)
                ids = (torch.arange(tokens * 16, device="cuda", dtype=torch.int32).view(tokens, 16)
                       + rank * tokens * 16).remainder(896)
                if routing == "hot_rank":
                    # All requests route to the same 16 experts on EP rank 0.
                    ids = torch.arange(16, device="cuda", dtype=torch.int32).expand(tokens, -1).contiguous()
                weights = torch.full((tokens, 16), 1 / 16, device="cuda", dtype=torch.float32)
                histogram = torch.bincount(ids.flatten().long(), minlength=896)
                dist.all_reduce(histogram, group=group)
                max_mean = (histogram.max().float() / histogram.float().mean()).item()
                dist.barrier()
                op = lambda: experts(x, ids, weights)
                samples = []
                for _ in range(3):
                    dist.barrier()
                    samples.append(worker.time(op, warmup=5, repeats=30))
                ms = sorted(samples)[1]
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        op()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with torch.cuda.graph(graph, stream=stream):
                    graph_result = op()
                graph_samples = []
                for _ in range(3):
                    dist.barrier()
                    graph_samples.append(worker.time(graph.replay, warmup=5, repeats=30))
                graph_ms = sorted(graph_samples)[1]
                del graph, graph_result
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                y = experts(x, ids, weights)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - base
                assert y.shape == x.shape and torch.count_nonzero(y).item() == 0
                rows.append(dict(rank=worker.rank, ep=ep, source_ranks=ep, tokens_per_rank=tokens,
                    global_tokens=ep * tokens, routing=routing, ms=ms, graph_ms=graph_ms,
                    max_mean_expert_load=max_mean,
                    workspace_device_bytes=workspace, dynamic_peak_bytes=peak))
                if worker.rank == 0:
                    print("MoE", rows[-1], flush=True)
                del x, ids, weights, histogram, y
        if ep == 16:
            for active in (4, 8):
                tokens = 8192 // active if rank < active else 0
                x = torch.zeros(tokens, 3584, device="cuda", dtype=torch.bfloat16)
                ids = (torch.arange(tokens * 16, device="cuda", dtype=torch.int32).view(tokens, 16)
                       + rank * tokens * 16).remainder(896)
                weights = torch.full((tokens, 16), 1 / 16, device="cuda", dtype=torch.float32)
                dist.barrier()
                op = lambda: experts(x, ids, weights)
                samples = []
                for _ in range(3):
                    dist.barrier()
                    samples.append(worker.time(op, warmup=5, repeats=30))
                ms = sorted(samples)[1]
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        op()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with torch.cuda.graph(graph, stream=stream):
                    graph_result = op()
                graph_samples = []
                for _ in range(3):
                    dist.barrier()
                    graph_samples.append(worker.time(graph.replay, warmup=5, repeats=30))
                graph_ms = sorted(graph_samples)[1]
                del graph, graph_result
                rows.append(dict(rank=worker.rank, ep=ep, source_ranks=active,
                    tokens_per_rank=tokens, global_tokens=8192, routing="balanced",
                    ms=ms, graph_ms=graph_ms, max_mean_expert_load=147 / (8192 * 16 / 896),
                    workspace_device_bytes=workspace, dynamic_peak_bytes=None))
                if worker.rank == 0:
                    print("MoE source placement", rows[-1], flush=True)
                del x, ids, weights
        del experts
        torch.cuda.empty_cache()
    return rows


@torch.inference_mode()
def measure_transition(worker):
    """End-to-end routed path including an optional, reversible owner redistribution."""
    import torch.distributed._symmetric_memory as symm_mem
    from dlengine.runtime.layers.backends.experts.mega_moe import MegaMoEExperts
    from dlengine.runtime.models.quant_config import QuantizationConfig
    from dlengine.runtime.runner.runner_config import set_runner_config

    symm_mem.set_backend('NCCL')
    group = worker.groups[16]
    rank = dist.get_rank(group)
    set_runner_config(mega_moe_max_tokens_per_rank=2048)
    quant = QuantizationConfig(format='mxfp4-pack-quantized',
        config_groups={'g': {'weights': {'group_size': 32}}})
    with torch.device('cuda'):
        experts = MegaMoEExperts(hidden_size=3584, intermediate_size=3072,
            num_experts=896, top_k=16, ep_size=16, tp_size=1,
            ep_group=group, quantization_config=quant)
    experts.gate_up_proj.zero_(); experts.down_proj.zero_()
    experts.gate_up_scale.fill_(127); experts.down_scale.fill_(127)
    experts.prepare_mega_weights(); experts._get_buffer()
    rows = []
    for total in (2048, 8192):
        for active in (4, 8, 16):
            local = total // active if rank < active else 0
            target = total // 16
            start = rank * (total // active)
            x = torch.full((local, 3584), float(rank + 1), device='cuda', dtype=torch.bfloat16)
            ids = (torch.arange(local * 16, device='cuda', dtype=torch.int32).view(local, 16)
                   + start * 16).remainder(896)
            weights = torch.full((local, 16), 1 / 16, device='cuda', dtype=torch.float32)
            outgoing = [max(0, min(start + local, (j + 1) * target) - max(start, j * target)) for j in range(16)]
            incoming = [max(0, min((j + 1) * (total // active), (rank + 1) * target)
                            - max(j * (total // active), rank * target)) if j < active else 0 for j in range(16)]
            balanced_x = torch.empty(target, 3584, device='cuda', dtype=torch.bfloat16)
            balanced_ids = torch.empty(target, 16, device='cuda', dtype=torch.int32)
            balanced_weights = torch.empty(target, 16, device='cuda', dtype=torch.float32)
            back = torch.empty_like(x)

            def distribute():
                for out, inp in [(balanced_x, x), (balanced_ids, ids), (balanced_weights, weights)]:
                    dist.all_to_all_single(out, inp, incoming, outgoing, group=group)

            def restore(value):
                dist.all_to_all_single(back, value, outgoing, incoming, group=group)
                return back

            distribute(); restore(balanced_x)
            torch.testing.assert_close(back, x, rtol=0, atol=0)
            expected_ids = (torch.arange(target * 16, device='cuda', dtype=torch.int32).view(target, 16)
                            + rank * target * 16).remainder(896)
            torch.testing.assert_close(balanced_ids, expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(balanced_weights, torch.full_like(balanced_weights, 1 / 16), rtol=0, atol=0)

            def with_transition():
                distribute()
                return restore(experts(balanced_x, balanced_ids, balanced_weights))

            operations = {'original_owners': lambda: experts(x, ids, weights),
                          'redistribute_routed_restore': with_transition,
                          'redistribute_restore_only': lambda: (distribute(), restore(balanced_x))}
            for name, op in operations.items():
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(5): op()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with torch.cuda.graph(graph, stream=stream):
                    graph_result = op()
                samples = []
                for _ in range(3):
                    dist.barrier()
                    samples.append(worker.time(graph.replay, warmup=5, repeats=30))
                rows.append(dict(rank=rank, global_tokens=total, source_ranks=active,
                                 operation=name, median_ms=sorted(samples)[1],
                                 min_ms=min(samples), max_ms=max(samples)))
                if rank == 0: print('TRANSITION', rows[-1], flush=True)
                del graph, graph_result
    return rows


@torch.inference_mode()
def measure_validation(worker):
    """Stress CP merge tails, noncontiguous LSE and fully masked partitions."""
    from bench.k3_partition.cp_merge import ContextMerger

    rows = []
    for cp in (2, 4, 8, 16):
        group = worker.groups[cp]
        for q in (1, 5, 33):
            for case in ('ordinary', 'large_lse', 'masked'):
                torch.manual_seed(9000 + worker.rank)
                out = torch.randn(q, 6, 128, device='cuda', dtype=torch.bfloat16)
                storage = torch.randn(6, q * 2, device='cuda', dtype=torch.float32)
                lse = storage[:, ::2]
                if case == 'large_lse':
                    lse.add_(10000 + 2 * (worker.rank % cp))
                if case == 'masked':
                    lse[0, 0] = -float('inf')
                    if worker.rank % cp == 0:
                        lse[1, :] = -float('inf')
                merger = ContextMerger(q, 6, 128, group)
                actual = merger(out, lse)
                gathered_out = [torch.empty_like(out) for _ in range(cp)]
                gathered_lse = [torch.empty_like(lse, memory_format=torch.contiguous_format) for _ in range(cp)]
                dist.all_gather(gathered_out, out, group=group)
                dist.all_gather(gathered_lse, lse.contiguous(), group=group)
                logits = torch.stack(gathered_lse).double()
                weights = logits.softmax(dim=0).nan_to_num(0)
                ref = (torch.stack(gathered_out).double() * weights.transpose(1, 2)[..., None]).sum(0)
                torch.testing.assert_close(actual.float(), ref.float(), atol=.016, rtol=.01)
                assert torch.isfinite(actual).all().item()
                error = (actual.float() - ref.float()).abs().max().item()
                rows.append(dict(rank=worker.rank, cp=cp, queries=q, case=case, max_abs_error=error))
    return rows
