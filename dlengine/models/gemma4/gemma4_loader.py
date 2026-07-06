import re
from typing import Generator, Tuple

import torch
from torch import nn

from dlengine.logging import get_logger
from dlengine.worker.loader import default_weight_loader

logger = get_logger()

_SKIP_PATTERNS = (
    "rotary_emb.",
    "vision_tower.",
    "audio_tower.",
    "multi_modal_projector.",
)


def load_weights(
    model: nn.Module,
    weights: Generator[Tuple[str, str, torch.Tensor], None, None],
) -> None:
    loaded_count = 0
    skipped_count = 0
    not_found_names: list[str] = []

    for weight_name, _raw_weight_name, tensor in weights:
        if weight_name.startswith("language_model."):
            weight_name = weight_name[len("language_model.") :]
        if any(pattern in weight_name for pattern in _SKIP_PATTERNS):
            skipped_count += 1
            continue

        try:
            param = model.get_parameter(weight_name)
        except AttributeError:
            not_found_names.append(weight_name)
            skipped_count += 1
            continue

        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, tensor)
        loaded_count += 1

    logger.warning(
        f"Gemma4 weight loading complete: {loaded_count} loaded, "
        f"{skipped_count} skipped"
    )
    if not_found_names:
        unique_patterns = set()
        for name in not_found_names:
            unique_patterns.add(re.sub(r"layers\.\d+\.", "layers.N.", name))
        logger.warning(
            f"  {len(not_found_names)} Gemma4 weights NOT FOUND "
            f"(unique patterns: {sorted(unique_patterns)})"
        )
