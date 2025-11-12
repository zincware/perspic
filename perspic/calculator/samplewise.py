from typing import Dict

import torch
import torch.func as func
import torch.nn as nn

from perspic.utils import set_track_running_stats


class SamplewiseCalculatorFunctorch:
    def __init__(
        self,
    ):
        self._track_running_stats = True

    def compute(
        self,
        model,
        loss_fn,
        inputs,
        targets,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute per-sample gradient norms for both the network parameters and
        the loss function using functorch.
        """
        # Set track_running_stats to False for BatchNorm layers to only update
        # on the current batch
        model = set_track_running_stats(model, False)
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

        # Restore the track_running_stats to True for BatchNorm layers
        # TODO: write test to verify this works correctly
        model = set_track_running_stats(model, track=True)
        return {
            "batch_grad_norms_network": batch_grad_norms_network,
            "batch_grad_norms_loss": batch_grad_norms_loss,
        }

    @staticmethod
    def _compute_per_sample_gradient_network_sum(model, inputs):
        """"""

        def model_fn(params, buffers, x):
            return func.functional_call(model, (params, buffers), x)  # .sum()

        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

        # grad_fn = func.grad(model_fn)
        grad_fn = func.jacrev(model_fn)
        per_sample_grads = func.vmap(grad_fn, in_dims=(None, None, 0))(
            params, buffers, inputs
        )

        return per_sample_grads

    @staticmethod
    def _compute_per_sample_gradient_norm_network(model, inputs):
        """
        Compute per-sample gradient magnitudes for the network function f(x)
        using functorch. -> ∇_theta f
        """
        # All samples simultaneously
        inputs = inputs.unsqueeze(1)  # Due to vmap
        per_sample_grads = SamplewiseCalculatorFunctorch._compute_per_sample_gradient_network_sum(
            model, inputs
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
        batch_grad_magnitudes = torch.sum(per_sample_grad_magnitudes, dim=0)

        return batch_grad_magnitudes

    @staticmethod
    def _compute_per_sample_grad_loss(
        model: nn.Module,
        loss_fn,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        def loss_fn_wrapped(outputs, targets):
            return loss_fn(outputs, targets)  # Sum ensures scalar loss

        outputs = model(inputs)
        # Compute per-sample gradient w.r.t. network outputs (∇_f L)
        grad_fn = func.jacrev(loss_fn_wrapped)
        per_sample_loss_grads = grad_fn(outputs, targets)

        return per_sample_loss_grads  # type: ignore

    @staticmethod
    def _compute_per_sample_gradient_norm_loss(model, loss_fn, inputs, targets):
        """
        Compute per-sample gradient magnitudes for the loss function L(f(x), y)
        using functorch. -> ∇_f L
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
        batch_grad_magnitudes = torch.sum(per_sample_grad_magnitudes, dim=0)

        return batch_grad_magnitudes
