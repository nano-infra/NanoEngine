import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn

from nanodeploy.models.weight_loadable import WeightLoadableModel
from nanodeploy.worker.loader import (
    load_deepseek_weights,
    load_model,
    should_load_deepseek_weight,
)


class _Projection(nn.Module):
    def __init__(
        self,
        weight_shape: tuple[int, ...],
        scale_shape: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(weight_shape))
        if scale_shape is not None:
            self.weight_scale_inv = nn.Parameter(torch.zeros(scale_shape))


class _MergedProjection(_Projection):
    def __init__(self) -> None:
        super().__init__((4, 2))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: int,
        weight_name: str,
    ) -> None:
        del weight_name
        offset = shard_id * loaded_weight.shape[0]
        param.data.narrow(0, offset, loaded_weight.shape[0]).copy_(loaded_weight)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fused_qkv_a_proj = _Projection((3, 2), (2, 1))
        self.kc = _Projection((2, 1, 2))
        self.vc = _Projection((2, 2, 1))


class _DenseMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = _Projection((4, 2), (2, 1))


class _Moe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.expert_list_this_rank = (2,)
        self.gate = nn.Module()
        self.gate.register_parameter(
            "e_score_correction_bias",
            nn.Parameter(torch.zeros(4, dtype=torch.float32)),
        )
        self.gate_up_proj = nn.Parameter(torch.zeros(1, 4, 2))
        self.gate_up_scale_inv = nn.Parameter(torch.zeros(1, 2, 1))
        self.down_proj = nn.Parameter(torch.zeros(1, 2, 2))
        self.down_scale_inv = nn.Parameter(torch.zeros(1, 1, 1))


class _Layer(nn.Module):
    def __init__(self, mlp: nn.Module, with_attention: bool = False) -> None:
        super().__init__()
        self.mlp = mlp
        if with_attention:
            self.self_attn = _Attention()


class _FakeDeepseek(WeightLoadableModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_Layer(_DenseMlp(), with_attention=True), _Layer(_Moe())]
        )
        self.lm_head = _Projection((2, 2))
        self.config = SimpleNamespace(
            num_attention_heads=2,
            qk_nope_head_dim=1,
            v_head_dim=1,
            kv_lora_rank=2,
        )
        self.quantization_config = SimpleNamespace(block_size=(2, 2))
        self._weight_loader_local_expert_indices: dict[str, dict[int, int]] = {}
        self.seen_weights: list[str] = []

    def should_load_weight(self, weight_name: str) -> bool:
        return should_load_deepseek_weight(self, weight_name)

    def load_weights(self, weights):
        def recording_weights():
            for name, tensor in weights:
                self.seen_weights.append(name)
                yield name, tensor

        return load_deepseek_weights(self, recording_weights())


class _DefaultMappedModel(WeightLoadableModel):
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = _MergedProjection()


class WeightLoaderTest(unittest.TestCase):
    def test_default_packed_mapping_remains_supported(self) -> None:
        model = _DefaultMappedModel()
        checkpoint = {
            "mlp.gate_proj.weight": torch.full((2, 2), 1.0),
            "mlp.up_proj.weight": torch.full((2, 2), 2.0),
        }

        with tempfile.TemporaryDirectory() as directory:
            save_file(checkpoint, str(Path(directory) / "model.safetensors"))
            load_model(model, directory)

        torch.testing.assert_close(
            model.mlp.gate_up_proj.weight,
            torch.tensor([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]),
        )

    def test_deepseek_v3_checkpoint_mapping(self) -> None:
        model = _FakeDeepseek()
        checkpoint = {
            "model.layers.0.self_attn.q_a_proj.weight": torch.full((2, 2), 1.0),
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": torch.full(
                (1, 2), 2.0
            ),
            "model.layers.0.self_attn.q_a_proj.weight_scale_inv": torch.full(
                (1, 1), 3.0
            ),
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv": (
                torch.full((1, 1), 4.0)
            ),
            "model.layers.0.mlp.gate_proj.weight": torch.full((2, 2), 5.0),
            "model.layers.0.mlp.up_proj.weight": torch.full((2, 2), 6.0),
            "model.layers.0.mlp.gate_proj.weight_scale_inv": torch.full(
                (1, 1), 7.0
            ),
            "model.layers.0.mlp.up_proj.weight_scale_inv": torch.full(
                (1, 1), 8.0
            ),
            "model.layers.0.self_attn.kv_b_proj.weight": torch.tensor(
                [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float8_e4m3fn
            ),
            "model.layers.0.self_attn.kv_b_proj.weight_scale_inv": torch.tensor(
                [[2.0], [3.0]]
            ),
            "model.layers.1.mlp.experts.2.gate_proj.weight": torch.full(
                (2, 2), 9.0
            ),
            "model.layers.1.mlp.experts.2.up_proj.weight": torch.full(
                (2, 2), 10.0
            ),
            "model.layers.1.mlp.experts.2.down_proj.weight": torch.full(
                (2, 2), 11.0
            ),
            "model.layers.1.mlp.experts.2.gate_proj.weight_scale_inv": torch.full(
                (1, 1), 12.0
            ),
            "model.layers.1.mlp.experts.2.up_proj.weight_scale_inv": torch.full(
                (1, 1), 13.0
            ),
            "model.layers.1.mlp.experts.2.down_proj.weight_scale_inv": torch.full(
                (1, 1), 14.0
            ),
            "model.layers.1.mlp.experts.1.gate_proj.weight": torch.full(
                (2, 2), 99.0
            ),
            "model.layers.1.mlp.gate.e_score_correction_bias": torch.full(
                (4,), 99.0
            ),
            "model.layers.0.self_attn.rotary_emb.inv_freq": torch.full(
                (2,), 99.0
            ),
            "model.layers.2.mlp.gate_proj.weight": torch.full((2, 2), 99.0),
            "lm_head.weight": torch.full((2, 2), 15.0),
        }

        with tempfile.TemporaryDirectory() as directory:
            save_file(checkpoint, str(Path(directory) / "model.safetensors"))
            load_model(model, directory)

        torch.testing.assert_close(
            model.model.layers[0].self_attn.fused_qkv_a_proj.weight,
            torch.tensor([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]]),
        )
        torch.testing.assert_close(
            model.model.layers[0].self_attn.fused_qkv_a_proj.weight_scale_inv,
            torch.tensor([[3.0], [4.0]]),
        )
        torch.testing.assert_close(
            model.model.layers[0].mlp.gate_up_proj.weight,
            torch.tensor(
                [[5.0, 5.0], [5.0, 5.0], [6.0, 6.0], [6.0, 6.0]]
            ),
        )
        torch.testing.assert_close(
            model.model.layers[0].mlp.gate_up_proj.weight_scale_inv,
            torch.tensor([[7.0], [8.0]]),
        )
        torch.testing.assert_close(
            model.model.layers[0].self_attn.kc.weight,
            torch.tensor([[[2.0, 4.0]], [[15.0, 18.0]]]),
        )
        torch.testing.assert_close(
            model.model.layers[0].self_attn.vc.weight,
            torch.tensor([[[6.0], [8.0]], [[21.0], [24.0]]]),
        )
        torch.testing.assert_close(
            model.model.layers[1].mlp.gate_up_proj[0],
            torch.tensor(
                [[9.0, 9.0], [9.0, 9.0], [10.0, 10.0], [10.0, 10.0]]
            ),
        )
        torch.testing.assert_close(
            model.model.layers[1].mlp.gate_up_scale_inv[0],
            torch.tensor([[12.0], [13.0]]),
        )
        torch.testing.assert_close(
            model.model.layers[1].mlp.down_proj[0],
            torch.full((2, 2), 11.0),
        )
        torch.testing.assert_close(
            model.model.layers[1].mlp.down_scale_inv[0],
            torch.tensor([[14.0]]),
        )
        torch.testing.assert_close(
            model.model.layers[1].mlp.gate.e_score_correction_bias,
            torch.full((4,), 99.0),
        )
        torch.testing.assert_close(model.lm_head.weight, torch.full((2, 2), 15.0))

        self.assertNotIn(
            "model.layers.1.mlp.experts.1.gate_proj.weight", model.seen_weights
        )
        self.assertIn(
            "model.layers.1.mlp.gate.e_score_correction_bias", model.seen_weights
        )
        self.assertNotIn(
            "model.layers.0.self_attn.rotary_emb.inv_freq", model.seen_weights
        )
        self.assertNotIn("model.layers.2.mlp.gate_proj.weight", model.seen_weights)


if __name__ == "__main__":
    unittest.main()
