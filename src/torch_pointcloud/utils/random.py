import logging
import random
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def seed_everything(seed: Optional[int] = None) -> int:
    """Set the seed for the random number generators in PyTorch, NumPy and Python.

    Args:
        seed: The seed to set for the random number generators. If None, a random seed will be selected.

    Returns:
        The seed that was set.
    """
    # Mostly copied from pytorch-lightning [1]
    # [1] https://github.com/Lightning-AI/pytorch-lightning/blob/1f5add327fd88fe288a2f889d720e5d5e06bd7d2/src/lightning/fabric/utilities/seed.py#L19
    if seed is None:
        seed = random.randint(min_seed_value, max_seed_value)

    log.info(f"Global seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def set_determinism(*, tf32: bool = False) -> None:
    """Set the TensorFloat-32 flags for fp32 CUDA matmul and cuDNN convolutions.

    TensorFloat-32 rounds fp32 matmul/convolution inputs to 19-bit mantissas on Ampere+ GPUs, which
    shifts benchmark metrics relative to references measured with it off. Neither the Lightning
    `Trainer(precision=...)` flag nor `torch.set_float32_matmul_precision` covers the cuDNN
    convolution path (`torch.backends.cudnn.allow_tf32` defaults to `True`), so both backends are
    pinned here. Nothing else is touched: RNG seeding is `seed_everything`, and deterministic
    kernel selection (`torch.use_deterministic_algorithms`,
    `torch.backends.cudnn.deterministic`) is not enabled.

    Args:
        tf32: Allow TensorFloat-32 in fp32 CUDA matmul and cuDNN convolutions.

    Example:
        >>> set_determinism(tf32=False)
        >>> torch.backends.cudnn.allow_tf32
        False
    """
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
