"""Base class for sample-wise gradient norm calculators."""

import warnings
from abc import ABC, abstractmethod
from typing import Callable, Dict

import torch
import torch.nn as nn


class SamplewiseCalculator(ABC):
    """Abstract base class for sample-wise gradient norm calculators.

    This class defines the interface for computing per-sample gradient norms.
    Subclasses must implement the abstract methods to provide specific
    implementations (e.g., using functorch, opacus, etc.).
    """

    @staticmethod
    def _warn_if_batchnorm_training(model: nn.Module) -> None:
        """Emit a warning if any BatchNorm layer is in training mode.

        BatchNorm layers in training mode use batch statistics that create
        coupling between samples, which breaks the per-sample gradient
        computation. This function warns users who may have forgotten to use
        the BatchStatSnapshot context manager.

        Args:
            model: The model to check for training-mode BatchNorm layers.
        """
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if module.training:
                    warnings.warn(
                        "BatchNorm layer detected in training mode. Per-sample gradient"
                        " norms may be incorrect due to batch statistics coupling. "
                        "Use the BatchStatSnapshot context manager to freeze batch "
                        "statistics for correct per-sample gradient computation.",
                        UserWarning,
                        stacklevel=4,
                    )
                    return  # Only warn once

    @staticmethod
    def compute_cross_metrics(
        sample_wise_metrics_self: Dict[str, torch.Tensor],
        sample_wise_metrics_cross: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute sample-wise cross metrics from self and cross batches.

        Computes the geometric mean of corresponding metrics from two batches:
        ``cross_metric = sqrt(metric_self * metric_cross)``. This provides a
        symmetric measure of gradient coupling between samples from different
        batches, useful for analyzing gradient interference during training.

        Args:
            sample_wise_metrics_self: Dictionary of sample-wise metrics (e.g.,
                gradient norms) computed from the training batch. Each value
                should be a tensor of shape (batch_size,).
            sample_wise_metrics_cross: Dictionary of sample-wise metrics computed
                from the cross batch. Must have the same keys as
                ``sample_wise_metrics_self``.

        Returns:
            Dictionary with the same keys as the inputs, where each value is
            the element-wise geometric mean of the corresponding input tensors.
        """
        cross_metrics = {}
        for key in sample_wise_metrics_self.keys():
            cross_metrics[key] = torch.sqrt(
                sample_wise_metrics_self[key] * sample_wise_metrics_cross[key]
            )
        return cross_metrics

    @abstractmethod
    def compute(
        self,
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
