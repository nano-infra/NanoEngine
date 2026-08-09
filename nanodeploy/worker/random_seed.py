import random

import numpy as np
import torch


def set_random_seed(seed: int | None) -> None:
    """Reset process RNGs in the same way as vLLM workers."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
