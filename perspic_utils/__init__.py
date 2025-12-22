"""
perspic_utils: Utility models and helpers for perspic.

This package provides:
- Pre-built model architectures (MLPs, CNNs)
- PyTorch Lightning training modules
- Model utilities and helpers
"""

# Import all models and modules from the models subpackage
from perspic_utils.models import (
    AdvancedClassificationModule,
    BatchNormMLP,
    ClassificationModule,
    ConfigurableMLP,
    DeepMLP,
    ResidualBlock,
    ResidualMLP,
    SimpleMLP,
    WideResNet,
)
from perspic_utils.models import print_model_info

__all__ = [
    # MLP models
    "SimpleMLP",
    "DeepMLP",
    "BatchNormMLP",
    "ConfigurableMLP",
    "ResidualMLP",
    "ResidualBlock",
    # Convolutional models
    "WideResNet",
    # Lightning modules
    "ClassificationModule",
    "AdvancedClassificationModule",
    # Utility functions
    "print_model_info",
]
