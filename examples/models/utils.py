"""Utility functions for model examples.

This module provides helper functions for inspecting and displaying
information about PyTorch models used in examples and tests.
"""


def print_model_info(model, verbose=True):
    """Print parameter count and optionally model architecture.

    Args:
        model: The PyTorch model to inspect
        verbose: If True, prints full architecture. If False, prints only name and parameter count.
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of trainable parameters: {num_params:,}")
    if verbose:
        print(f"Architecture:\n{model}")
    print("-" * 50)
