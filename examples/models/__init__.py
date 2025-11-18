"""
Example models package for CIFAR-10 and testing.

This package provides:
- Pure PyTorch MLP architectures (mlps.py)
- Lightning wrapper modules for training (lightning_modules.py)
"""

# Import Lightning modules
from .lightning_modules import (
    AdvancedClassificationModule,
    ClassificationModule,
)

# Import MLP models
from .mlps import (
    BatchNormMLP,
    ConfigurableMLP,
    DeepMLP,
    ResidualBlock,
    ResidualMLP,
    SimpleMLP,
)

__all__ = [
    # MLP models
    "SimpleMLP",
    "DeepMLP",
    "BatchNormMLP",
    "ConfigurableMLP",
    "ResidualMLP",
    "ResidualBlock",
    # Lightning modules
    "ClassificationModule",
    "AdvancedClassificationModule",
]
