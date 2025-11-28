"""Base class for sample-wise gradient norm calculators."""

from abc import ABC, abstractmethod
from typing import Callable, Dict

import torch
import torch.nn as nn


class SamplewiseCalculator(ABC):
    """Abstract base class for sample-wise gradient norm calculators.

    This class defines the interface for computing per-sample gradient norms.
    Subclasses must implement the abstract methods to provide specific
    implementations (e.g., using functorch, opacus, etc.).

    All methods are static as they don't require instance state.
    """

    __slots__ = ()  # No instance attributes needed

    @staticmethod
    @abstractmethod
    def compute(
        model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-sample gradient norms.

        Args:
            model: The neural network model.
            loss_fn: Loss function callable that takes (predictions, targets)
                and returns a scalar loss tensor.
            inputs: Input tensor batch of shape (batch_size, ...).
            targets: Target tensor batch of shape (batch_size, ...).

        Returns:
            Dictionary containing:
                - 'batch_grad_norms_network': Gradient norms for network parameters.
                - 'batch_grad_norms_loss': Gradient norms for the loss function.
        """
        ...

    @staticmethod
    @abstractmethod
    def _compute_per_sample_gradient_norm_network(
        model: nn.Module, inputs: torch.Tensor, reduce: bool = True
    ) -> torch.Tensor:
        """Compute per-sample gradient norms for network parameters.

        Args:
            model: The neural network model.
            inputs: Input tensor batch of shape (batch_size, ...).
            reduce: If True, sum over batch dimension. If False, return
                per-sample squared norms.

        Returns:
            If reduce=True: Scalar tensor (sum of squared gradient norms).
            If reduce=False: Tensor of shape (batch_size,) with per-sample
                squared gradient norms.
        """
        ...

    @staticmethod
    @abstractmethod
    def _compute_per_sample_gradient_norm_loss(
        model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        inputs: torch.Tensor,
        targets: torch.Tensor,
        reduce: bool = True,
    ) -> torch.Tensor:
        """Compute per-sample gradient norms for the loss function.

        Args:
            model: The neural network model.
            loss_fn: Loss function callable that takes (predictions, targets)
                and returns a scalar loss tensor.
            inputs: Input tensor batch of shape (batch_size, ...).
            targets: Target tensor batch of shape (batch_size, ...).
            reduce: If True, sum over batch dimension. If False, return
                per-sample squared norms.

        Returns:
            If reduce=True: Scalar tensor (sum of squared gradient norms).
            If reduce=False: Tensor of shape (batch_size,) with per-sample
                squared gradient norms.
        """
        ...
