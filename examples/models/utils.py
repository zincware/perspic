"""Utility functions for models.

This module contains utility functions for all model-related components,
including both pure PyTorch architectures and PyTorch Lightning modules.
"""

import torch.nn as nn


def print_model_info(model: nn.Module, verbose: bool = True) -> None:
    """Print parameter count and optionally model architecture.

    Note: This function is intended for pure PyTorch architectures (nn.Module),
    not for PyTorch Lightning modules.

    Args:
        model: The PyTorch model to inspect.
        verbose: If True, prints full architecture. If False, prints only name and parameter count.
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of trainable parameters: {num_params:,}")
    if verbose:
        print(f"Architecture:\n{model}")
    print("-" * 50)
