from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import nanodeploy.models.deepseek_v2 as deepseek_module
from nanodeploy.models.deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2MoE,
    deepseek_grouped_topk,
)


class _RecordingSharedExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.input = hidden_states.detach().clone()
        return torch.zeros_like(hidden_states)


class _FakeFusedMoe:
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        return hidden_states + 10.0


class _ForwardOnlyDeepseekMoE(DeepseekV2MoE):
    @property
    def expert_list_this_rank(self) -> list[int]:
        return [0]

    def fusedmoe_build(self, low_latency_mode: bool) -> _FakeFusedMoe:
        del low_latency_mode
        return self.fake_moe


class _RecordingRotaryEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k_pe_shape: tuple[int, ...] | None = None

    def forward(
        self,
        positions: torch.Tensor,
        q_pe: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del positions
        self.k_pe_shape = tuple(k_pe.shape)
        return q_pe + 10.0, k_pe + 20.0


class _ProjectOnlyDeepseekAttention(DeepseekV2Attention):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.num_heads = 2
        self.kv_lora_rank = 3
        self.rotary_emb = _RecordingRotaryEmbedding()

    def _qkv_proj(self, hidden_states: torch.Tensor, num_heads: int):
        num_tokens = hidden_states.shape[0]
        assert num_heads == self.num_heads
        query_states = torch.zeros(num_tokens, num_heads, 5)
        key_states = torch.zeros(num_tokens, 5)
        value_states = torch.arange(num_tokens * 3, dtype=torch.float32).reshape(
            num_tokens, 3
        )
        q_pe = torch.arange(
            num_tokens * num_heads * 2, dtype=torch.float32
        ).reshape(num_tokens, num_heads, 2)
        k_pe = torch.arange(num_tokens * 2, dtype=torch.float32).reshape(
            num_tokens, 2
        )
        return query_states, key_states, value_states, q_pe, k_pe


def _reference_deepseek_v3_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.sigmoid(router_logits.float())
    choice_scores = scores + correction_bias.unsqueeze(0)
    num_tokens = router_logits.shape[0]
    group_scores = (
        choice_scores.view(num_tokens, 8, 32)
        .topk(2, dim=-1, sorted=False)
        .values.sum(dim=-1)
    )
    selected_groups = group_scores.topk(4, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, 8, 32)
        .reshape(num_tokens, 256)
    )
    selected_experts = choice_scores.masked_fill(
        ~expert_mask, float("-inf")
    ).topk(8, dim=-1, sorted=True).indices
    routing_weights = scores.gather(1, selected_experts)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights *= 2.5
    return routing_weights, selected_experts


def test_deepseek_v3_grouped_topk_matches_reference() -> None:
    generator = torch.Generator().manual_seed(17)
    router_logits = torch.randn(5, 256, generator=generator).to(torch.bfloat16)
    correction_bias = torch.randn(256, generator=generator) * 0.15

    actual_weights, actual_experts = deepseek_grouped_topk(
        router_logits,
        top_k=8,
        num_expert_group=8,
        topk_group=4,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=2.5,
        e_score_correction_bias=correction_bias,
        sorted_topk=True,
    )
    expected_weights, expected_experts = _reference_deepseek_v3_topk(
        router_logits, correction_bias
    )

    torch.testing.assert_close(actual_experts, expected_experts)
    torch.testing.assert_close(actual_weights, expected_weights)
    assert actual_weights.dtype == torch.float32
    torch.testing.assert_close(
        actual_weights.sum(dim=-1),
        torch.full((5,), 2.5),
    )


def test_correction_bias_changes_selection_but_not_mixture_weights() -> None:
    router_logits = torch.tensor(
        [[4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0]]
    )
    correction_bias = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 9.0]
    )

    routing_weights, selected_experts = deepseek_grouped_topk(
        router_logits,
        top_k=2,
        num_expert_group=4,
        topk_group=2,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=2.5,
        e_score_correction_bias=correction_bias,
        sorted_topk=True,
    )

    torch.testing.assert_close(selected_experts, torch.tensor([[6, 7]]))
    expected = torch.sigmoid(router_logits[:, [6, 7]])
    expected = expected / expected.sum(dim=-1, keepdim=True) * 2.5
    torch.testing.assert_close(routing_weights, expected)


def test_shared_expert_receives_original_hidden_states() -> None:
    moe = _ForwardOnlyDeepseekMoE.__new__(_ForwardOnlyDeepseekMoE)
    nn.Module.__init__(moe)
    moe.ep_size = 2
    moe.num_experts = 4
    moe.top_k = 2
    moe.quantization_config = SimpleNamespace(quant_method="fp8")
    moe.gate = nn.Linear(2, 4, bias=False)
    moe.gate_up_proj = None
    moe.gate_up_scale_inv = None
    moe.down_proj = None
    moe.down_scale_inv = None
    moe.fake_moe = _FakeFusedMoe()
    moe.shared_experts = _RecordingSharedExpert()
    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    with (
        patch.object(
            deepseek_module,
            "get_context",
            return_value=SimpleNamespace(is_prefill=True),
        ),
        patch.object(
            deepseek_module,
            "get_runner_config",
            return_value=SimpleNamespace(
                moe_routing_simulation_strategy="uniform_random"
            ),
        ),
    ):
        output = moe(hidden_states)

    torch.testing.assert_close(moe.shared_experts.input, hidden_states)
    torch.testing.assert_close(output, hidden_states + 10.0)


def test_rotary_components_are_written_back_to_attention_projections() -> None:
    attention = _ProjectOnlyDeepseekAttention()

    query_states, key_states, value_states = attention.project_for_attention(
        torch.tensor([0, 1]),
        torch.zeros(2, 4),
    )

    assert attention.rotary_emb.k_pe_shape == (2, 1, 2)
    torch.testing.assert_close(
        query_states[..., 3:],
        torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) + 10.0,
    )
    torch.testing.assert_close(
        key_states[..., 3:],
        torch.arange(4, dtype=torch.float32).reshape(2, 1, 2) + 20.0,
    )
    torch.testing.assert_close(
        value_states,
        torch.arange(6, dtype=torch.float32).reshape(2, 1, 3),
    )
