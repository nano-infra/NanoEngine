"""Quick test: does flash_attn_3 flash_attn_varlen_func handle seqlen_q != seqlen_k with causal=True correctly?

The expected behavior (same as flash_attn 2) is bottom-right aligned causal mask:
  Q[j] can attend to K[0 .. seqlen_k - seqlen_q + j]

This is critical for chunked prefill where we process a chunk of Q tokens
against the full KV context.
"""

import torch


def reference_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, causal, scale):
    """Pure PyTorch reference attention with varlen + causal support."""
    num_seqs = len(cu_seqlens_q) - 1
    outputs = []
    for i in range(num_seqs):
        q_start, q_end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
        k_start, k_end = cu_seqlens_k[i], cu_seqlens_k[i + 1]
        qi = q[q_start:q_end]  # [sq, h, d]
        ki = k[k_start:k_end]  # [sk, h, d]
        vi = v[k_start:k_end]  # [sk, h, d]
        sq, sk = qi.shape[0], ki.shape[0]

        # [h, sq, sk]
        scores = torch.einsum("qhd,khd->hqk", qi.float(), ki.float()) * scale

        if causal:
            # Bottom-right aligned causal mask:
            # Q[j] attends to K[0 .. sk - sq + j]
            row_idx = torch.arange(sq, device=q.device).unsqueeze(1)  # [sq, 1]
            col_idx = torch.arange(sk, device=q.device).unsqueeze(0)  # [1, sk]
            mask = col_idx <= (row_idx + sk - sq)  # [sq, sk]
            scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("hqk,khd->qhd", attn, vi.float())
        outputs.append(out)

    return torch.cat(outputs, dim=0).to(q.dtype)


def test_flash_attn_varlen_causal_qk_mismatch():
    from flash_attn_interface import flash_attn_varlen_func

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    num_heads = 4
    head_dim = 64
    seqlen_q = 32
    seqlen_k = 128
    scale = head_dim**-0.5

    q = torch.randn(seqlen_q, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(seqlen_k, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(seqlen_k, num_heads, head_dim, device=device, dtype=dtype)

    cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, seqlen_k], dtype=torch.int32, device=device)

    ref_out = reference_attention(
        q, k, v, cu_seqlens_q.tolist(), cu_seqlens_k.tolist(), causal=True, scale=scale
    )

    fa3_out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        softmax_scale=scale,
        causal=True,
    )

    diff = (fa3_out.float() - ref_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_norm = ref_out.float().norm().item()

    print(
        f"\n=== flash_attn_varlen_func causal test (seqlen_q={seqlen_q}, seqlen_k={seqlen_k}) ==="
    )
    print(f"  ref output norm:  {ref_norm:.4f}")
    print(f"  fa3 output norm:  {fa3_out.float().norm().item():.4f}")
    print(f"  max abs diff:     {max_diff:.6f}")
    print(f"  mean abs diff:    {mean_diff:.6f}")
    print(
        f"  relative diff:    {diff.sum().item() / ref_out.float().abs().sum().item():.6f}"
    )

    if max_diff < 0.05:
        print(
            "  PASS: flash_attn_varlen_func handles seqlen_q < seqlen_k with causal=True correctly"
        )
    else:
        print(
            "  FAIL: flash_attn_varlen_func does NOT correctly handle seqlen_q < seqlen_k with causal=True!"
        )
        print("  This is likely the root cause of the chunked prefill bug.")

        # Also check: does it match a TOP-LEFT aligned causal mask? (Q[j] attends to K[0..j])
        scores_tl = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
        row_idx = torch.arange(seqlen_q, device=device).unsqueeze(1)
        col_idx = torch.arange(seqlen_k, device=device).unsqueeze(0)
        mask_tl = col_idx <= row_idx
        scores_tl = scores_tl.masked_fill(~mask_tl.unsqueeze(0), float("-inf"))
        attn_tl = torch.softmax(scores_tl, dim=-1)
        ref_tl = torch.einsum("hqk,khd->qhd", attn_tl, v.float()).to(dtype)
        diff_tl = (fa3_out.float() - ref_tl.float()).abs()
        print(f"  Top-left causal max diff: {diff_tl.max().item():.6f}")
        if diff_tl.max().item() < 0.05:
            print(
                "  -> FA3 uses TOP-LEFT aligned causal mask (Q[j] attends to K[0..j])"
            )
            print("  -> This is WRONG for chunked prefill!")


def test_flash_attn_varlen_causal_equal_lens():
    """Sanity check: when seqlen_q == seqlen_k, causal should be standard."""
    from flash_attn_interface import flash_attn_varlen_func

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    num_heads = 4
    head_dim = 64
    seqlen = 64
    scale = head_dim**-0.5

    q = torch.randn(seqlen, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(seqlen, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(seqlen, num_heads, head_dim, device=device, dtype=dtype)

    cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32, device=device)

    ref_out = reference_attention(
        q, k, v, cu_seqlens.tolist(), cu_seqlens.tolist(), causal=True, scale=scale
    )
    fa3_out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        softmax_scale=scale,
        causal=True,
    )
    diff = (fa3_out.float() - ref_out.float()).abs()
    print(f"\n=== Sanity check (seqlen_q == seqlen_k == {seqlen}) ===")
    print(f"  max abs diff: {diff.max().item():.6f}")
    print(f"  PASS" if diff.max().item() < 0.05 else "  FAIL")


if __name__ == "__main__":
    test_flash_attn_varlen_causal_equal_lens()
    test_flash_attn_varlen_causal_qk_mismatch()
