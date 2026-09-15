import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_chunked_prefix_mla_matches_unsplit_attention():
    from flash_attn.cute import flash_attn_varlen_func

    from dlengine.runtime.layers.backends.attention.mla_utils import (
        chunked_prefix_mla_attention,
    )

    torch.manual_seed(7)
    tq, tc, heads, latent, nope, rope, value = 16, 48, 2, 32, 16, 8, 16
    q = torch.randn(tq, heads, nope + rope, device="cuda", dtype=torch.bfloat16)
    fresh_latent = torch.randn(tq, latent, device="cuda", dtype=torch.bfloat16)
    cached = torch.randn(tc, latent + rope, device="cuda", dtype=torch.bfloat16)
    kc = torch.randn(latent, heads * nope, device="cuda", dtype=torch.bfloat16)
    vc = torch.randn(latent, heads * value, device="cuda", dtype=torch.bfloat16)
    kf = torch.cat(
        [
            (fresh_latent @ kc).view(tq, heads, nope),
            torch.zeros(tq, heads, rope, device="cuda", dtype=torch.bfloat16),
        ],
        -1,
    )
    vf = (fresh_latent @ vc).view(tq, heads, value)
    cuq = torch.tensor([0, tq], device="cuda", dtype=torch.int32)
    split = chunked_prefix_mla_attention(
        q,
        kf,
        vf,
        cached,
        torch.tensor([tc], device="cuda"),
        cuq,
        kc,
        vc,
        chunk_size=16,
        softmax_scale=(nope + rope) ** -0.5,
        attention_func=flash_attn_varlen_func,
    )
    kc_full = torch.cat(
        [
            (cached[:, :latent] @ kc).view(tc, heads, nope),
            cached[:, None, latent:].expand(-1, heads, -1),
        ],
        -1,
    )
    vc_full = (cached[:, :latent] @ vc).view(tc, heads, value)
    k = torch.cat([kc_full, kf])
    v = torch.cat([vc_full, vf])
    cuk = torch.tensor([0, tc + tq], device="cuda", dtype=torch.int32)
    ref = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cuq,
        cu_seqlens_k=cuk,
        max_seqlen_q=tq,
        max_seqlen_k=tc + tq,
        causal=True,
        softmax_scale=(nope + rope) ** -0.5,
    )
    ref = ref[0] if isinstance(ref, tuple) else ref
    torch.testing.assert_close(split, ref, atol=3e-2, rtol=3e-2)
