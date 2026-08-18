from collections.abc import Iterable
from typing import ClassVar

import torch
from torch import nn


def default_weight_loader(param: nn.Parameter, tensor: torch.Tensor) -> None:
    param.data.copy_(tensor)


class WeightLoadableModel(nn.Module):
    """Explicit checkpoint-loading contract for NanoDeploy models."""

    packed_modules_mapping: ClassVar[dict[str, tuple[str, int | str]]] = {}

    def should_load_weight(self, weight_name: str) -> bool:
        return True

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        for weight_name, loaded_weight in weights:
            for checkpoint_name, (
                param_name,
                shard_id,
            ) in self.packed_modules_mapping.items():
                if checkpoint_name not in weight_name:
                    continue
                mapped_name = weight_name.replace(checkpoint_name, param_name)
                param = self.get_parameter(mapped_name)
                param.weight_loader(
                    param,
                    loaded_weight,
                    shard_id,
                    weight_name,
                )
                break
            else:
                param = self.get_parameter(weight_name)
                weight_loader = getattr(
                    param,
                    "weight_loader",
                    default_weight_loader,
                )
                weight_loader(param, loaded_weight)
