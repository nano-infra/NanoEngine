import pytest
import torch
from dlengine.runtime.layers.backends.megamoe import MegaMoEExperts
from dlengine.runtime.layers.blackwell import BlackwellBackendFactory
from dlengine.runtime.models.quant_config import QuantizationConfig


def mxfp4_config():
    return QuantizationConfig(
        format="mxfp4-pack-quantized",
        config_groups={"g": {"weights": {"group_size": 32}}},
    )


def test_blackwell_maps_mxfp4_experts_to_megamoe():
    factory = BlackwellBackendFactory(mxfp4_config())
    experts = factory.get_distributed_routed_experts(
        hidden_size=64,
        intermediate_size=32,
        num_experts=8,
        top_k=2,
        ep_size=2,
        tp_size=1,
    )
    assert isinstance(experts, MegaMoEExperts)
    assert experts.gate_up_proj.shape == (4, 64, 32)
    assert experts.down_proj.shape == (4, 64, 16)
    assert experts.gate_up_scale.shape == (4, 64, 2)
    assert experts.down_scale.shape == (4, 64, 1)


def test_megamoe_rejects_ffn_tp_instead_of_padding_batch():
    with pytest.raises(ValueError, match="requires ffn_tp=1"):
        MegaMoEExperts(
            hidden_size=64,
            intermediate_size=32,
            num_experts=8,
            top_k=2,
            ep_size=2,
            tp_size=2,
            quantization_config=mxfp4_config(),
        )


def test_megamoe_requires_ep():
    with pytest.raises(ValueError, match="ffn_ep > 1"):
        MegaMoEExperts(
            hidden_size=64,
            intermediate_size=32,
            num_experts=8,
            top_k=2,
            ep_size=1,
            tp_size=1,
            quantization_config=mxfp4_config(),
        )


def test_megamoe_forward_never_falls_back_before_weight_prepare():
    experts = MegaMoEExperts(
        hidden_size=64,
        intermediate_size=32,
        num_experts=8,
        top_k=2,
        ep_size=2,
        tp_size=1,
        quantization_config=mxfp4_config(),
    )
    with pytest.raises(RuntimeError, match="weights are not prepared"):
        experts(
            torch.empty(1, 64, dtype=torch.bfloat16),
            torch.zeros(1, 2, dtype=torch.int32),
            torch.ones(1, 2, dtype=torch.float32),
            is_prefill=False,
        )


def test_megamoe_requires_k3_situ_constants():
    common = dict(
        hidden_size=64,
        intermediate_size=32,
        num_experts=8,
        top_k=2,
        ep_size=2,
        tp_size=1,
        quantization_config=mxfp4_config(),
    )
    with pytest.raises(ValueError, match="activation='situ'"):
        MegaMoEExperts(**common, activation="silu")
    with pytest.raises(ValueError, match="beta=4"):
        MegaMoEExperts(**common, activation_situ_beta=3.0)


def test_k3_packed_expert_loader_uses_ep_only_and_merges_w1_w3():
    experts = MegaMoEExperts(
        hidden_size=64,
        intermediate_size=32,
        num_experts=8,
        top_k=2,
        ep_size=2,
        tp_size=1,
        quantization_config=mxfp4_config(),
    )
    w1 = torch.full((32, 32), 11, dtype=torch.uint8)
    w3 = torch.full((32, 32), 33, dtype=torch.uint8)
    w2 = torch.full((64, 16), 22, dtype=torch.uint8)
    s1 = torch.full((32, 2), 1, dtype=torch.uint8)
    s3 = torch.full((32, 2), 3, dtype=torch.uint8)
    s2 = torch.full((64, 1), 2, dtype=torch.uint8)

    # EP rank 1 owns global experts 4..7; global expert 4 maps to local 0.
    for projection, kind, tensor in (
        ("w1", "weight_packed", w1),
        ("w3", "weight_packed", w3),
        ("w2", "weight_packed", w2),
        ("w1", "weight_scale", s1),
        ("w3", "weight_scale", s3),
        ("w2", "weight_scale", s2),
    ):
        assert experts.load_expert_weight(4, projection, kind, tensor, ep_rank=1)

    assert torch.equal(experts.gate_up_proj[0, :32], w1)
    assert torch.equal(experts.gate_up_proj[0, 32:], w3)
    assert torch.equal(experts.down_proj[0], w2)
    assert torch.equal(experts.gate_up_scale[0, :32], s1)
    assert torch.equal(experts.gate_up_scale[0, 32:], s3)
    assert torch.equal(experts.down_scale[0], s2)


def test_k3_packed_expert_loader_rejects_bad_local_shape():
    experts = MegaMoEExperts(
        hidden_size=64,
        intermediate_size=32,
        num_experts=8,
        top_k=2,
        ep_size=2,
        tp_size=1,
        quantization_config=mxfp4_config(),
    )
    with pytest.raises(ValueError, match="expected"):
        experts.load_expert_weight(
            0,
            "w1",
            "weight_packed",
            torch.empty(31, 32, dtype=torch.uint8),
            ep_rank=0,
        )
