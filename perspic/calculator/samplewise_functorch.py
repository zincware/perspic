from typing import Callable, Dict

import torch
import torch.func as func
import torch.nn as nn

from perspic.calculator.samplewise import SamplewiseCalculator


class SamplewiseCalculatorFunctorch(SamplewiseCalculator):
    """Calculate per-sample gradient norms using functorch."""

    @staticmethod
    def compute(
        model: nn.Module,
        loss_fn: Callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute per-sample gradient norms for both the network parameters and
        the loss function using functorch.

        Args:
            model: The neural network model. The model's forward is assumed to support
                sample-wise gradient computation. For models containing BatchNorm
                layers, see e.g. `BatchStatSnapshot` context manager in `perspic.utils`
                to temporarily adjust the BatchNorm behavior for correct sample-wise
                gradient computation.
            loss_fn: Loss function callable.
            inputs: Input tensor batch.
            targets: Target tensor batch.

        Returns:
            Dictionary containing batch gradient norms for network and loss.
        """
        batch_grad_norms_network = (
            SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_network(
                model, inputs
            )
        )
        batch_grad_norms_loss = (
            SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_loss(
                model, loss_fn, inputs, targets
            )
        )

        # Restore the track_running_states to True for BatchNorm layers
        return {
            "batch_grad_norms_network": batch_grad_norms_network,
            "batch_grad_norms_loss": batch_grad_norms_loss,
        }

    @staticmethod
    def _compute_per_sample_gradient_network_sum(
        model: nn.Module, inputs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute per-sample gradients for network parameters.

        Args:
            model: The neural network model.
            inputs: Input tensor batch.

        Returns:
            Dictionary mapping parameter names to per-sample gradients.
        """

        def model_fn(params, buffers, x):
            return func.functional_call(model, (params, buffers), x)

        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

        grad_fn = func.jacrev(model_fn)
        per_sample_grads = func.vmap(grad_fn, in_dims=(None, None, 0))(
            params, buffers, inputs
        )

        return per_sample_grads

    @staticmethod
    def _compute_per_sample_gradient_norm_network(
        model: nn.Module, inputs: torch.Tensor, reduce: bool = True
    ) -> torch.Tensor:
        """
        Compute per-sample gradient magnitudes for the network function f(x)
        using functorch. -> ∇_theta f

        Args:
            model: The neural network model.
            inputs: Input tensor batch.
            reduce: If True, sum over batch. If False, return per-sample norms.

        Returns:
            If reduce=True: Scalar (sum of squared gradient norms).
            If reduce=False: Tensor (batch_size,) with per-sample squared norms.
        """
        # All samples simultaneously
        inputs = inputs.unsqueeze(1)  # Due to vmap
        per_sample_grads = (
            SamplewiseCalculatorFunctorch._compute_per_sample_gradient_network_sum(
                model, inputs
            )
        )
        # Assert the correct shape of the gradients
        # (batch_size, ...)
        params = dict(model.named_parameters())
        for k, v in per_sample_grads.items():
            # Assert that the first dimension is the batch size
            assert v.shape[0] == inputs.shape[0]
            # Assert that the v.shape[1:] matches the shape of the parameter
            assert v.shape[-len(params[k].shape) :] == params[k].shape
        # Compute per-sample gradient magnitude (L2 norm)
        per_sample_grad_magnitudes = torch.stack(
            [(g**2).sum(dim=tuple(range(1, g.ndim))) for g in per_sample_grads.values()]
        ).sum(
            dim=0
        )  # Sum across parameters

        return (
            per_sample_grad_magnitudes.sum() if reduce else per_sample_grad_magnitudes
        )

    @staticmethod
    def _compute_per_sample_grad_loss(
        model: nn.Module,
        loss_fn: Callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample gradients of loss w.r.t. network outputs.

        Args:
            model: The neural network model.
            loss_fn: Loss function callable.
            inputs: Input tensor batch.
            targets: Target tensor batch.

        Returns:
            Per-sample loss gradients tensor.
        """

        def loss_fn_wrapped(outputs, targets):
            return loss_fn(outputs, targets)

        outputs = model(inputs)
        # Compute per-sample gradient w.r.t. network outputs (∇_f L)
        grad_fn = func.jacrev(loss_fn_wrapped)
        per_sample_loss_grads = grad_fn(outputs, targets)

        return per_sample_loss_grads  # type: ignore

    @staticmethod
    def _compute_per_sample_gradient_norm_loss(
        model: nn.Module,
        loss_fn: Callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        reduce: bool = True,
    ) -> torch.Tensor:
        """
        Compute per-sample gradient magnitudes for the loss function L(f(x), y)
        using functorch. -> ∇_f L

        Args:
            model: The neural network model.
            loss_fn: Loss function callable.
            inputs: Input tensor batch.
            targets: Target tensor batch.
            reduce: If True, sum over batch. If False, return per-sample norms.

        Returns:
            If reduce=True: Scalar (sum of squared gradient norms).
            If reduce=False: Tensor (batch_size,) with per-sample squared norms.
        """
        per_sample_loss_grads = (
            SamplewiseCalculatorFunctorch._compute_per_sample_grad_loss(
                model, loss_fn, inputs, targets
            )
        )
        # Assert the correct shape of the gradients
        # (batch_size, ...)
        assert per_sample_loss_grads.shape[0] == inputs.shape[0]
        # Compute per-sample gradient magnitude (L2 norm)
        per_sample_grad_magnitudes = (per_sample_loss_grads**2).sum(
            dim=tuple(range(1, per_sample_loss_grads.ndim))
        )

        return (
            per_sample_grad_magnitudes.sum() if reduce else per_sample_grad_magnitudes
        )
