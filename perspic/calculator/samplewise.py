"""Base class for sample-wise gradient norm calculators."""

import warnings
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

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
    def resolve_target_mask(
        targets: torch.Tensor, ignore_index: Optional[int]
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Determine which target positions are real vs. ignored.

        Shared by both calculator backends so "should this batch be masked"
        is decided in exactly one place -- the Opacus and functorch
        implementations must never disagree about it, since `compute()`
        divides `batch_grad_norms_network` and multiplies
        `batch_grad_norms_loss` by the *same* `n_elements`, and only that
        agreement makes the two normalizations cancel in
        ``CouplingCalculator`` (chi_loss * chi_net is independent of
        `n_elements` -- see `coupling.py`).

        Args:
            targets: Target tensor of shape (batch, ...), e.g. (B,) for plain
                classification or (B, T) for a sequence model's per-position
                class indices.
            ignore_index: Target value marking a position that contributes no
                loss and should be excluded from the sample-wise projection
                (mirrors `nn.CrossEntropyLoss`'s `ignore_index`, default
                -100). Pass `None` to disable masking entirely regardless of
                the batch's contents. `targets` must line up with the model
                output's leading axes ((B, T) targets for (B, T, V) output;
                a (B, V, T) output layout is not supported). Criteria that
                shift labels internally (HF-style causal LM) must be given
                already-shifted targets here, otherwise the mask is off by
                one position.

        Returns:
            A `(mask, n_elements)` tuple. `mask` is `None` when no masking
            applies -- `ignore_index` is `None`, `targets` is float/complex
            (one-hot/soft targets; see `compute()`'s `normalize=False` for
            those), or the batch simply contains no `ignore_index` value.
            Returning `None` rather than an all-True tensor is what keeps an
            unmasked batch's computation bitwise identical to before this
            feature existed -- callers must skip the masking multiply
            entirely when `mask is None`, not multiply by an all-True mask.
            When masking does apply, `mask` is a boolean tensor shaped like
            `targets` (True = real, scored position). `n_elements` is
            `targets.numel()` when `mask is None`, else `mask.sum()`.

        Warns:
            UserWarning: If every position is `ignore_index`. The loss then
                has no real positions, so chi_net/chi_loss for this batch are
                NaN/0 (the return values are unchanged).
        """
        if ignore_index is None:
            return None, targets.numel()
        if targets.is_floating_point() or targets.is_complex():
            return None, targets.numel()
        is_ignored = targets.eq(ignore_index)
        if not bool(is_ignored.any()):
            return None, targets.numel()
        mask = ~is_ignored
        n_elements = mask.sum()
        if n_elements == 0:
            warnings.warn(
                "Every target position equals ignore_index="
                f"{ignore_index}; the loss has no real positions, so "
                "chi_net/chi_loss will be NaN/0 for this batch.",
                UserWarning,
                stacklevel=2,
            )
        return mask, n_elements

    @staticmethod
    def broadcast_mask(mask: torch.Tensor, ndim: int) -> torch.Tensor:
        """Append trailing singleton axes so `mask` broadcasts against a
        tensor of `ndim` dimensions.

        `mask` covers every leading axis a label tensor has (e.g. (B, T) for
        a sequence model's per-position targets); the tensor it must
        broadcast against carries additional trailing axes the label doesn't
        have -- at minimum the loss-reduction ("class"/"vocab") axis, and for
        functorch's per-sample-gradient tensors, the parameter axes on top of
        that. This is a plain reshape (`unsqueeze`), not a copy; broadcasting
        multiplication (`*`) handles the rest without materializing the
        expanded size.
        """
        while mask.dim() < ndim:
            mask = mask.unsqueeze(-1)
        return mask

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
        model: nn.Module,
        inputs: torch.Tensor,
        reduce: bool = True,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-sample gradient norms for network parameters.

        Args:
            model: The neural network model.
            inputs: Input tensor batch of shape (batch_size, ...).
            reduce: If True, sum over batch dimension. If False, return
                per-sample squared norms.
            mask: Optional boolean tensor shaped like a label tensor (e.g.
                (batch, seq_len) for a sequence model), True at positions to
                include in the projection. See `resolve_target_mask`/
                `broadcast_mask`. `None` (the default) computes over every
                output position, exactly as before this parameter existed.

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
