from dlengine.models.gemma4.gemma4_loader import _is_unused_shared_kv_weight
from torch import nn


class _Attention(nn.Module):
    def __init__(self, shared: bool):
        super().__init__()
        self.is_kv_shared_layer = shared


class _Layer(nn.Module):
    def __init__(self, shared: bool):
        super().__init__()
        self.self_attn = _Attention(shared)


class _Gemma(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(False), _Layer(True)])


def test_only_expected_shared_kv_weights_are_skipped():
    model = _Gemma()
    assert _is_unused_shared_kv_weight(model, "model.layers.1.self_attn.k_proj.weight")
    assert _is_unused_shared_kv_weight(model, "model.layers.1.self_attn.v_proj.weight")
    assert _is_unused_shared_kv_weight(model, "model.layers.1.self_attn.k_norm.weight")
    assert not _is_unused_shared_kv_weight(
        model, "model.layers.0.self_attn.k_proj.weight"
    )
    assert not _is_unused_shared_kv_weight(
        model, "model.layers.1.self_attn.q_proj.weight"
    )
