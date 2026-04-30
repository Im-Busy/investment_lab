"""
GPU utility functions for device management and tensor operations.
"""

from typing import Optional

import numpy as np
import torch


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(
    arr: np.ndarray,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert numpy array to torch tensor on specified device.

    Args:
        arr: NumPy array to convert
        device: Target device (defaults to best available)
        dtype: Target dtype

    Returns:
        Torch tensor on specified device
    """
    if device is None:
        device = get_device()
    tensor = torch.as_tensor(arr, dtype=dtype)
    return tensor.to(device)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert torch tensor to numpy array.

    Handles GPU tensors by moving to CPU first.

    Args:
        tensor: Torch tensor (on any device)

    Returns:
        NumPy array on CPU
    """
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return tensor.numpy()
