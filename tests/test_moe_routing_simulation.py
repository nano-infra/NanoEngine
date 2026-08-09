import random

import numpy as np
import torch

from nanodeploy.worker.random_seed import set_random_seed
from nanodeploy.worker.runner_config import (
    get_runner_config,
    reset_runner_config,
    set_runner_config,
)


def test_runner_config_propagates_moe_routing_simulation_settings():
    try:
        set_runner_config(
            max_num_seqs=32,
            dummy_weight=True,
            perfect_eplb=False,
            moe_routing_simulation_strategy="uniform_random",
            seed=0,
        )

        config = get_runner_config()
        assert config.moe_routing_simulation_strategy == "uniform_random"
        assert config.seed == 0
    finally:
        reset_runner_config()


def test_set_random_seed_matches_vllm_seed_targets(monkeypatch):
    calls = []
    monkeypatch.setattr(random, "seed", lambda seed: calls.append(("random", seed)))
    monkeypatch.setattr(np.random, "seed", lambda seed: calls.append(("numpy", seed)))
    monkeypatch.setattr(torch, "manual_seed", lambda seed: calls.append(("torch", seed)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: calls.append(("cuda", seed)),
    )

    set_random_seed(0)

    assert calls == [
        ("random", 0),
        ("numpy", 0),
        ("torch", 0),
        ("cuda", 0),
    ]


def test_set_random_seed_accepts_none(monkeypatch):
    monkeypatch.setattr(
        random,
        "seed",
        lambda seed: (_ for _ in ()).throw(AssertionError(seed)),
    )

    set_random_seed(None)
