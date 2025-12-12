"""Sample-wise gradient norm calculator using Opacus with ghost clipping."""

from typing import Callable, Dict

import torch
import torch.nn as nn
from opacus.grad_sample import register_grad_sampler, register_norm_sampler
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)
from opacus.utils.module_utils import requires_grad, trainable_parameters

from perspic.calculator.samplewise import SamplewiseCalculator


# Register BatchNorm gradient samplers for eval mode (frozen statistics)
@register_grad_sampler(nn.BatchNorm1d)
@register_grad_sampler(nn.BatchNorm2d)
@register_grad_sampler(nn.BatchNorm3d)
def _compute_batch_norm_grad_sample(
    layer: nn.modules.batchnorm._BatchNorm,
    activations: list[torch.Tensor],
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """Compute per-sample gradients for BatchNorm layers in eval mode."""
    activations = activations[0]
    mean = layer.running_mean
    var = layer.running_var
    eps = layer.eps

    # Reshape for broadcasting: [C] -> [1, C, 1, 1, ...]
    view_shape = [1, layer.num_features] + [1] * (activations.dim() - 2)
    mean = mean.view(view_shape)
    var = var.view(view_shape)

    # Normalize: x_hat = (x - mu) / sqrt(var + eps)
    x_hat = (activations - mean) / torch.sqrt(var + eps)

    # Sum over spatial dimensions (all except batch and channel)
    sum_dims = list(range(2, activations.dim()))

    if sum_dims:
        grad_weight = torch.sum(backprops * x_hat, dim=sum_dims)
        grad_bias = torch.sum(backprops, dim=sum_dims)
    else:
        grad_weight = backprops * x_hat
        grad_bias = backprops

    ret = {}
    if layer.weight is not None:
        ret[layer.weight] = grad_weight
    if layer.bias is not None:
        ret[layer.bias] = grad_bias

    return ret


@register_norm_sampler(nn.BatchNorm1d)
@register_norm_sampler(nn.BatchNorm2d)
@register_norm_sampler(nn.BatchNorm3d)
def _compute_batch_norm_norm_sample(
    layer: nn.modules.batchnorm._BatchNorm,
    activations: list[torch.Tensor],
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """Compute per-sample gradient norms for BatchNorm layers."""
    grads = _compute_batch_norm_grad_sample(layer, activations, backprops)
    return {param: grad.norm(2, dim=1) for param, grad in grads.items()}


class _GhostNormFastGradientClipping(GradSampleModuleFastGradientClipping):
    """Fast gradient clipping module that works in eval mode (frozen batch stats)."""

    def capture_activations_hook(
        self,
        module: nn.Module,
        forward_input: list[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
        if (
            not requires_grad(module)
            or not torch.is_grad_enabled()
            or not self.hooks_enabled
        ):
            return

        if not hasattr(module, "activations"):
            module.activations = []
        module.activations.append([t.detach() for t in forward_input])

        for _, p in trainable_parameters(module):
            p._forward_counter += 1
            if (
                self.use_ghost_clipping
                and p._forward_counter > 1
                and type(module) in self.NORM_SAMPLERS
            ):
                raise NotImplementedError(
                    "Parameter tying is not supported with Ghost Clipping"
                )


def _disable_inplace_ops(model: nn.Module) -> dict[nn.Module, bool]:
    """Temporarily disable in-place operations in activation modules.

    Opacus's backward hooks create views of tensors, which conflict with
    in-place operations (e.g., ReLU(inplace=True)). This function disables
    in-place operations and returns the original states for restoration.

    Args:
        model: The model to modify.

    Returns:
        Dictionary mapping modules to their original `inplace` values.
    """
    original_states: dict[nn.Module, bool] = {}
    for module in model.modules():
        if hasattr(module, "inplace") and module.inplace:
            original_states[module] = True
            module.inplace = False
    return original_states


def _restore_inplace_ops(original_states: dict[nn.Module, bool]) -> None:
    """Restore original in-place operation states.

    Args:
        original_states: Dictionary from `_disable_inplace_ops`.
    """
    for module, inplace in original_states.items():
        module.inplace = inplace


def _cleanup_opacus_leftovers(model: nn.Module) -> None:
    """Remove leftover attributes from Opacus ghost clipping.

    Opacus's `remove_hooks()` only removes the forward/backward hooks but
    leaves behind attributes on modules and parameters that can interfere
    with subsequent training or analysis.

    Args:
        model: The model to clean up.
    """
    for module in model.modules():
        if hasattr(module, "activations"):
            delattr(module, "activations")
    for param in model.parameters():
        if hasattr(param, "_forward_counter"):
            delattr(param, "_forward_counter")
        if hasattr(param, "grad_sample"):
            delattr(param, "grad_sample")
        if hasattr(param, "norm_sample"):
            delattr(param, "norm_sample")
        if hasattr(param, "summed_grad"):
            delattr(param, "summed_grad")


def _reset_opacus_state(model: nn.Module) -> None:
    """Reset Opacus state for reusing the wrapper between backward passes.

    This clears activations and resets forward counters without removing hooks,
    allowing efficient reuse of the GradSampleModule wrapper across multiple reuses.

    Args:
        model: The model wrapped by Opacus.
    """
    for module in model.modules():
        if hasattr(module, "activations"):
            module.activations = []
    for param in model.parameters():
        if hasattr(param, "_forward_counter"):
            param._forward_counter = 0
        if hasattr(param, "_norm_sample"):
            delattr(param, "_norm_sample")


def _draw_rademacher_random_vector(
    shape: tuple[int, ...], device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Draw a random vector from the Rademacher distribution (entries are +1 or -1)."""
    return torch.randint(0, 2, shape, device=device, dtype=dtype) * 2 - 1


class SamplewiseCalculatorOpacus(SamplewiseCalculator):
    """Calculate per-sample gradient norms using Opacus with ghost clipping.

    This implementation uses Opacus's efficient per-sample gradient norm
    computation with support for BatchNorm layers in eval mode.

    Args:
        strict: If True, Opacus will validate that all layers are supported
            for per-sample gradient computation. Defaults to False.
        approximate_with_n: If not None, use Hutchinson's trace estimator with n
            random projections instead of iterating over all output dimensions.
            This provides a faster but approximate computation. Defaults to None
            (exact computation).

    Note:
        For models with BatchNorm, wrap calls with `BatchStatSnapshot` context
        manager to freeze running statistics, similar to the functorch calculator.
    """

    def __init__(self, strict: bool = False, approximate_with_n: int | None = None):
        self.strict = strict
        self.approximate_with_n = approximate_with_n

    def compute(
        self,
        model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        inputs: torch.Tensor,
        targets: torch.Tensor,
        normalize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-sample gradient norms for network and loss.

        Args:
            model: The neural network model.
            loss_fn: Loss function callable that takes (predictions, targets)
                and returns a scalar loss tensor.
            inputs: Input tensor batch of shape (batch_size, ...).
            targets: Target tensor batch of shape (batch_size, ...).
            normalize: If True, sample-wise metrics are correctd to scale properly with
                batch-size.

        Returns:
            Dictionary with 'batch_grad_norms_network' and 'batch_grad_norms_loss'.
        """
        batch_grad_norms_network = (
            SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
                model,
                inputs,
                strict=self.strict,
                approximate_with_n=self.approximate_with_n,
            )
        )
        batch_grad_norms_loss = (
            SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_loss(
                model, loss_fn, inputs, targets
            )
        )

        # Optionally normalize the results
        if normalize:
            batch_size = inputs.shape[0]
            batch_grad_norms_network /= batch_size
            batch_grad_norms_loss *= batch_size

        model.zero_grad()
        return {
            "batch_grad_norms_network": batch_grad_norms_network,
            "batch_grad_norms_loss": batch_grad_norms_loss,
        }

    @staticmethod
    def _compute_per_sample_gradient_norm_network(
        model: nn.Module,
        inputs: torch.Tensor,
        reduce: bool = True,
        strict: bool = False,
        approximate_with_n: int | None = None,
    ) -> torch.Tensor:
        """Compute per-sample gradient norms for network parameters (∇_θ f).

        Uses ghost clipping to compute ||∇_θ f(x_i)||² by iterating over
        output dimensions. The Opacus wrapper is created once and reused across
        output dimensions for efficiency.

        Args:
            model: The neural network model.
            inputs: Input tensor batch of shape (batch_size, ...).
            reduce: If True, sum over batch. If False, return per-sample norms.
            strict: If True, Opacus validates all layers are supported.
            approximate_with_n: If not None, the sample-wise gradients will not be
                compute for each output dimension. Instead, we will apply n
                low-dimensional projections to estimate the sum of output dimensions.

        Returns:
            If reduce=True: Scalar (sum of squared gradient norms).
            If reduce=False: Tensor (batch_size,) with per-sample squared norms.
        """
        SamplewiseCalculatorOpacus._warn_if_batchnorm_training(model)

        # Temporarily disable in-place operations (incompatible with Opacus hooks)
        inplace_states = _disable_inplace_ops(model)

        try:
            # Determine output dimension
            with torch.no_grad():
                sample_out = model(inputs[:1])
            output_dim = sample_out.shape[-1] if sample_out.dim() > 1 else 1

            total_sq_norms = torch.zeros(inputs.shape[0], device=inputs.device)

            # Create wrapper once and reuse across output dimensions
            gs_model = _GhostNormFastGradientClipping(
                model, strict=strict, loss_reduction="sum"
            )

            if approximate_with_n is not None:
                # Implementation of Hutchinson's Trace estimator
                # Each iteration requires a fresh forward pass because Opacus
                # consumes activations during backward.
                vectors = _draw_rademacher_random_vector(
                    shape=(approximate_with_n, output_dim), device=inputs.device
                )

                for v in vectors:
                    # Reset Opacus state for fresh forward/backward pass
                    _reset_opacus_state(model)

                    gs_model.zero_grad()
                    out = gs_model(inputs)
                    projected = (out * v).sum()
                    projected.backward()

                    total_sq_norms += gs_model.get_norm_sample() ** 2

            else:
                for dim in range(output_dim):
                    # Reset Opacus state for fresh forward/backward pass
                    _reset_opacus_state(model)

                    gs_model.zero_grad()
                    out = gs_model(inputs)[:, dim]
                    out.sum().backward()

                    total_sq_norms += gs_model.get_norm_sample() ** 2

            gs_model.remove_hooks()

            # Clean up leftover Opacus attributes from the original model
            _cleanup_opacus_leftovers(model)

            return total_sq_norms.sum() if reduce else total_sq_norms
        finally:
            # Restore original in-place operation states
            _restore_inplace_ops(inplace_states)

    @staticmethod
    def _compute_per_sample_gradient_norm_loss(
        model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        inputs: torch.Tensor,
        targets: torch.Tensor,
        reduce: bool = True,
    ) -> torch.Tensor:
        """Compute per-sample gradient norms for the loss function (∇_f L).

        Args:
            model: The neural network model.
            loss_fn: Loss function callable that takes (predictions, targets)
                and returns a scalar loss tensor.
            inputs: Input tensor batch of shape (batch_size, ...).
            targets: Target tensor batch of shape (batch_size, ...).
            reduce: If True, sum over batch. If False, return per-sample norms.

        Returns:
            If reduce=True: Scalar (sum of squared gradient norms).
            If reduce=False: Tensor (batch_size,) with per-sample squared norms.
        """
        SamplewiseCalculatorOpacus._warn_if_batchnorm_training(model)

        outputs = model(inputs)
        outputs = outputs.detach().requires_grad_(True)

        loss = loss_fn(outputs, targets)
        (grad_outputs,) = torch.autograd.grad(loss, outputs)

        # Compute squared norms
        per_sample_sq_norms = (grad_outputs**2).sum(
            dim=tuple(range(1, grad_outputs.ndim))
        )
        return per_sample_sq_norms.sum() if reduce else per_sample_sq_norms
