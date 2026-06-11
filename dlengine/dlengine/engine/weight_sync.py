"""Engine-side weight sync entry point.

Thin facade over ``RayExecutor.collective_rpc`` that fans an
``apply_weight_update`` call out to every worker. Designed to be called
from outside the engine (e.g. by an RL training loop) but works just as
well in process for tests.

Kept separate from ``llm_engine.py`` so future drain / quiesce / async
semantics live next to each other rather than scattered through the
engine's main loop.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("dlengine")


def update_weights(executor, named_tensors: dict[str, torch.Tensor]):
    """Broadcast ``named_tensors`` to every worker.

    Each worker resolves its own TP/EP slice via the per-parameter
    ``weight_loader`` callback (see
    ``dlengine.context.weight.apply_named_tensors_in_place``).

    Args:
        executor: An object exposing ``collective_rpc(method, args)`` —
            typically ``LLMEngine.executor`` (a ``RayExecutor`` or
            ``DLSlimeExecutor``).
        named_tensors: HF-named full tensors. Tensors not on this rank's
            parameter set are silently skipped.

    Returns:
        ``list[dict]`` — one stats dict per worker (see
        ``apply_named_tensors_in_place`` for the schema).
    """
    counts = executor.collective_rpc("apply_weight_update", (named_tensors,))
    logger.info("engine.update_weights -> %s", counts)
    return counts
