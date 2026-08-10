"""Global Random Seed Management for Complete Reproducibility.

Applies seed across Python random, NumPy, PyTorch CPU & CUDA backends,
and configures CuDNN deterministic operations.
"""

import os
import random

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False


def set_global_seed(seed: int = 42) -> None:
    """Set global random seed across all random number generators.

    Args:
        seed: Random seed integer.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[Reproducibility] Set global random seed to: {seed}")
