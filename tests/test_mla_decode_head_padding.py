"""Regression: K3's 24/96 local MLA heads retain their outputs after padding."""
import pytest
import torch

from dlengine.runtime.context.batch import reset_batch_context, set_batch_context


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason='Blackwell TRTLLM-GEN required',
)
@pytest.mark.parametrize('heads', [24, 96])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float8_e4m3fn])
@torch.inference_mode()
def test_padded_mla_heads_match_dense_reference_and_graph(heads, dtype):
    from dlengine.runtime.layers.backends.mla.trtllm import TrtllmMlaAttention

    batch, length, pages = 32, 133, 4
    torch.manual_seed(561)
    q = torch.randn(batch, heads, 576, device='cuda', dtype=torch.bfloat16)
    cache = (torch.randn(batch * pages, 64, 1, 576, device='cuda', dtype=torch.bfloat16) * .2).to(dtype)
    table = torch.arange(batch * pages - 1, -1, -1, device='cuda', dtype=torch.int32).reshape(batch, pages)
    layer = TrtllmMlaAttention(heads, 576, 192 ** -.5, 1, 512,
        mla_qk_nope_head_dim=128, mla_kv_lora_rank=512)
    layer.k_cache = cache
    set_batch_context(is_prefill=False, block_tables=table.unsqueeze(0),
        context_lens=torch.full((1, batch), length, device='cuda', dtype=torch.int32))
    try:
        actual = layer(q, None, None, write_kv_cache=False)
        kv = cache.float()[table.long()].reshape(batch, pages * 64, 576)[:, :length]
        qref = q.to(dtype).float()
        scores = torch.einsum('bhd,bkd->bhk', qref, kv) * 192 ** -.5
        reference = torch.einsum('bhk,bkv->bhv', scores.softmax(-1), kv[..., :512])
        # Match the native FP8 attention tolerance used by the CP reference
        # checks; a norm bound also protects against broad systematic error.
        difference = actual.float() - reference
        relative_l2 = difference.norm() / reference.norm()
        assert relative_l2 < (.035 if dtype == torch.float8_e4m3fn else .006)
        torch.testing.assert_close(actual.float(), reference,
            atol=.003 if dtype == torch.float8_e4m3fn else .002, rtol=.06)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(q, None, None, write_kv_cache=False)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, actual, atol=0, rtol=0)
    finally:
        reset_batch_context()
