"""
Multi-Layer Perceptron (MLP) models for CIFAR-10 and testing purposes.

This module contains pure PyTorch implementations of various MLP architectures
for use in examples and tests. All models are designed for CIFAR-10 (32x32x3 images, 10 classes).
"""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    A simple single hidden layer MLP for CIFAR-10.

    Architecture: Input (3072) -> Hidden (512) -> Output (10)
    """

    def __init__(self, width=128):
        super().__init__()

        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(32 * 32 * 3, width), nn.ReLU(), nn.Linear(width, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)


class DeepMLP(nn.Module):
    """
    A deeper MLP with multiple hidden layers and dropout for CIFAR-10.

    Architecture: Input (3072) -> 1024 -> 512 -> 256 -> 128 -> Output (10)
    Includes dropout for regularization.
    """

    def __init__(self, dropout=0.3):
        super().__init__()

        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(32 * 32 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)


class BatchNormMLP(nn.Module):
    """
    MLP with Batch Normalization layers for improved training stability.

    Architecture: Input (3072) -> 512 -> BN -> 256 -> BN -> Output (10)
    """

    def __init__(self, width1=128, width2=128):
        super().__init__()

        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(32 * 32 * 3, width1),
            nn.BatchNorm1d(width1),
            nn.ReLU(),
            nn.Linear(width1, width2),
            nn.BatchNorm1d(width2),
            nn.ReLU(),
            nn.Linear(width2, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)


class ConfigurableMLP(nn.Module):
    """
    A configurable MLP with customizable architecture.

    Includes:
    - Configurable hidden layer sizes
    - Optional batch normalization
    - Optional dropout
    """

    def __init__(self, hidden_sizes=[1024, 512, 256], dropout=0.2, use_batch_norm=True):
        super().__init__()

        # Build the network dynamically
        layers = [nn.Flatten()]
        input_size = 32 * 32 * 3

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(input_size, 10))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ResidualMLP(nn.Module):
    """
    MLP with residual connections (skip connections) for better gradient flow.

    Similar to ResNet but with fully connected layers instead of convolutions.
    """

    def __init__(self, hidden_size=512, num_blocks=3):
        super().__init__()

        self.flatten = nn.Flatten()

        # Initial projection
        self.input_proj = nn.Linear(32 * 32 * 3, hidden_size)

        # Residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_size) for _ in range(num_blocks)]
        )

        # Output layer
        self.output = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        return self.output(x)


class ResidualBlock(nn.Module):
    """A residual block for the ResidualMLP."""

    def __init__(self, hidden_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


# Utility function to print model info
def print_model_info(model):
    """Print parameter count and model architecture."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of trainable parameters: {num_params:,}")
    print(f"Architecture:\n{model}")
    print("-" * 50)


if __name__ == "__main__":
    # Example usage: print info for all models
    print("=" * 50)
    print("MLP Models Overview")
    print("=" * 50)

    models = [
        SimpleMLP(),
        BatchNormMLP(),
        DeepMLP(),
        ConfigurableMLP(),
        ResidualMLP(),
    ]

    for model in models:
        print_model_info(model)
