from types import SimpleNamespace

import pytest
import torch
from dlengine.runtime.layers.base_backend import PrequantizedActivation
from dlengine.runtime.layers.rotary_embedding import RotaryEmbedding
from dlengine.runtime.models.deepseek_v2 import deepseek_v2


def _bare_attention() -> deepseek_v2.DeepseekV2Attention:
    attention = deepseek_v2.DeepseekV2Attention.__new__(deepseek_v2.DeepseekV2Attention)
    torch.nn.Module.__init__(attention)
    return attention


def test_shared_projection_input_quantizes_once(monkeypatch):
    attention = _bare_attention()
    attention._share_qkv_input_quant = True
    attention._input_quant_round_ue8m0 = False
    attention.kv_a_proj_with_mqa = SimpleNamespace(
        weight=torch.empty(1, dtype=torch.float8_e4m3fn)
    )
    calls = []

    def fake_quant(x, group_size, *, dtype, round_ue8m0):
        calls.append((x, group_size, dtype, round_ue8m0))
        return (
            torch.empty((128, x.shape[1]), dtype=dtype),
            torch.empty((128, x.shape[1] // group_size), dtype=torch.float32),
        )

    monkeypatch.setattr(deepseek_v2, "quant_fp8_tma", fake_quant)
    hidden_states = torch.randn(6, 6144, dtype=torch.bfloat16)

    result = attention._shared_projection_input(hidden_states)

    assert isinstance(result, PrequantizedActivation)
    assert result.num_tokens == 6
    assert len(calls) == 1
    assert calls[0][1:] == (128, torch.float8_e4m3fn, False)


def test_shared_projection_input_preserves_fallback_tensor():
    attention = _bare_attention()
    attention._share_qkv_input_quant = False
    hidden_states = torch.randn(6, 6144, dtype=torch.bfloat16)

    assert attention._shared_projection_input(hidden_states) is hidden_states


def test_kv_proj_preserves_eager_fallback(monkeypatch):
    attention = _bare_attention()
    attention.kv_lora_rank = 4
    attention._inplace_mla_kv_norm = True
    projected = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    attention.kv_a_proj_with_mqa = lambda _hidden: projected.clone()
    attention.kv_a_layernorm = torch.nn.RMSNorm(4, eps=1e-6)
    monkeypatch.setattr(
        deepseek_v2, "can_use_strided_inplace_rms_norm", lambda *_args: False
    )

    key_states, value_states, k_pe = attention._kv_proj(torch.empty(3, 1))

    expected = attention.kv_a_layernorm(projected[:, :4])
    torch.testing.assert_close(value_states, expected)
    torch.testing.assert_close(key_states[:, :4], expected)
    torch.testing.assert_close(k_pe, projected[:, 4:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kv_proj_uses_inplace_strided_rms_norm():
    attention = _bare_attention()
    attention.kv_lora_rank = 512
    attention._inplace_mla_kv_norm = True
    projected = torch.randn(6, 576, dtype=torch.bfloat16, device="cuda")
    attention.kv_a_proj_with_mqa = lambda _hidden: projected.clone()
    attention.kv_a_layernorm = deepseek_v2.RMSNorm(512, eps=1e-6).cuda().bfloat16()
    attention.kv_a_layernorm.requires_grad_(False)

    with torch.inference_mode():
        reference = attention.kv_a_layernorm(projected[:, :512].contiguous())
        key_states, value_states, k_pe = attention._kv_proj(
            torch.empty(6, 1, dtype=torch.bfloat16, device="cuda")
        )

    assert value_states.data_ptr() == key_states.data_ptr()
    assert value_states.stride() == (576, 1)
    torch.testing.assert_close(value_states, reference, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k_pe, projected[:, 512:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_mla_qk_rope_matches_half_layout_reference():
    if deepseek_v2._FUSED_MLA_QK_ROPE is None:
        pytest.skip("vendored fused RoPE is unavailable")

    device = torch.device("cuda")
    rotary = RotaryEmbedding(64, 64, 512, 10000).to(device)
    positions = torch.tensor([0, 1, 17, 255], dtype=torch.int64, device=device)
    q_parent = torch.randn(4, 8, 192, dtype=torch.bfloat16, device=device)
    k_parent = torch.randn(4, 1, 576, dtype=torch.bfloat16, device=device)
    q_interleaved = q_parent[..., -64:]
    k_interleaved = k_parent[..., -64:]

    q_reference = deepseek_v2._interleaved_to_half(q_interleaved.clone())
    k_reference = deepseek_v2._interleaved_to_half(k_interleaved.clone())
    q_reference, k_reference = rotary(positions, q_reference, k_reference)

    fused = deepseek_v2._apply_fused_mla_qk_rope(
        rotary, positions, q_interleaved, k_interleaved
    )
    assert fused is not None
    q_fused, k_fused = fused
    torch.testing.assert_close(
        q_fused,
        q_reference,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        k_fused,
        k_reference,
        rtol=0,
        atol=0,
    )
