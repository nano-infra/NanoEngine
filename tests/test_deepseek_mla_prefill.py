from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import nanodeploy.models.deepseek_v2 as deepseek_module
from nanodeploy.models.deepseek_v2 import DeepseekV2Attention, DeepseekV2BMM


class _AdditiveRotary(nn.Module):
    def forward(self, positions, q_pe, k_pe):
        del positions
        return q_pe + 10.0, k_pe + 20.0


class _RecordingNativeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: tuple[torch.Tensor, ...] | None = None

    def forward_mla_prefill_native(self, q, k, v, cache_k):
        self.inputs = tuple(tensor.detach().clone() for tensor in (q, k, v, cache_k))
        return v


class _NativePrefillAttention(DeepseekV2Attention):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.num_heads = 2
        self.q_head_dim = 3
        self.qk_nope_head_dim = 2
        self.qk_rope_head_dim = 1
        self.kv_lora_rank = 3
        self.v_head_dim = 2
        self.kc = DeepseekV2BMM(2, 2, 3)
        self.vc = DeepseekV2BMM(2, 3, 2)
        self.rotary_emb = _AdditiveRotary()
        self.attn_fwd = _RecordingNativeAttention()
        self.o_proj = nn.Identity()

        self.q = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        )
        self.compressed_kv = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.k_pe = torch.tensor([[0.5], [1.5]])

        self.kc.weight.data.copy_(
            torch.tensor(
                [
                    [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]],
                    [[2.0, 1.0, 0.0], [3.0, 0.0, 1.0]],
                ]
            )
        )
        self.vc.weight.data.copy_(
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                    [[2.0, 0.0], [0.0, 2.0], [1.0, -1.0]],
                ]
            )
        )

    def _qkv_proj_native(self, hidden_states, num_heads):
        assert hidden_states.shape == (2, 4)
        assert num_heads == self.num_heads
        q = self.q.clone()
        key_states = torch.cat([self.compressed_kv, self.k_pe], dim=-1)
        return q, key_states, self.compressed_kv, q[..., 2:], self.k_pe


def test_deepseek_prefill_expands_native_keys_and_values_before_attention():
    attention = _NativePrefillAttention()
    context = SimpleNamespace(is_prefill=True, prefill_has_prefix=False)

    with patch.object(deepseek_module, "get_context", return_value=context):
        output = attention(
            torch.tensor([0, 1]),
            torch.zeros(2, 4),
        )

    q, k, v, cache_k = attention.attn_fwd.inputs
    expected_q = attention.q.clone()
    expected_q[..., 2:] += 10.0
    expected_k_nope = torch.einsum(
        "tr,hdr->thd", attention.compressed_kv, attention.kc.weight
    )
    expected_k = torch.cat(
        [
            expected_k_nope,
            (attention.k_pe + 20.0).unsqueeze(1).expand(-1, 2, -1),
        ],
        dim=-1,
    )
    expected_v = torch.einsum(
        "tr,hrv->thv", attention.compressed_kv, attention.vc.weight
    )
    expected_cache_k = torch.cat(
        [attention.compressed_kv, attention.k_pe + 20.0], dim=-1
    ).unsqueeze(1)

    torch.testing.assert_close(q, expected_q)
    torch.testing.assert_close(k, expected_k)
    torch.testing.assert_close(v, expected_v)
    torch.testing.assert_close(cache_k, expected_cache_k)
    torch.testing.assert_close(output, expected_v.reshape(2, -1))
