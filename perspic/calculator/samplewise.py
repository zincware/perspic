import torch.nn as nn
import torch
import torch.func as func
from typing import Dict


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
        self.model = self.set_track_running_stats(model, False)
        batch_grad_norms_network = (
            self._compute_per_sample_gradient_norm_network(model, inputs)
        )
        batch_grad_norms_loss = (
            self._compute_per_sample_gradient_norm_loss(
                model, loss_fn, inputs, targets
            )
        )

        # Restore the track_running_stats to True for BatchNorm layers
        # TODO: write test to verify this works correctly
        self.model = self.set_track_running_stats(self.model, track=True)
        return {
            "batch_grad_norms_network": batch_grad_norms_network,
            "batch_grad_norms_loss": batch_grad_norms_loss,
        }

    def _compute_per_sample_gradient_network_sum(self, model, inputs):
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

    def _compute_per_sample_gradient_norm_network(self, model, inputs):
        """
        Compute per-sample gradient magnitudes for the network function f(x)
        using functorch. -> ∇_theta f
        """
        # All samples simultaneously
        inputs = inputs.unsqueeze(1)  # Due to vmap
        per_sample_grads = self._compute_per_sample_gradient_network_sum(
            model, inputs
        )
        # Assert the correct shape of the gradients
        # (batch_size, ...)
        params = dict(model.named_parameters())
        for k, v in per_sample_grads.items():
            # Assert that the first dimension is the batch size
            assert v.shape[0] == inputs.shape[0]
            # Assert that the v.shape[1:] matches the shape of the parameter
            assert v.shape[-len(params[k].shape):] == params[k].shape

        # Compute per-sample gradient magnitude (L2 norm)
        per_sample_grad_magnitudes = torch.stack(
            [
                (g**2).sum(dim=tuple(range(1, g.ndim)))
                for g in per_sample_grads.values()
            ]
        ).sum(
            dim=0
        )  # Sum across parameters
        batch_grad_magnitudes = torch.sum(per_sample_grad_magnitudes, dim=0)

        return batch_grad_magnitudes

    def _compute_per_sample_grad_loss(
        self,
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

    def _compute_per_sample_gradient_norm_loss(
        self, model, loss_fn, inputs, targets
    ):
        """
        Compute per-sample gradient magnitudes for the loss function L(f(x), y)
        using functorch. -> ∇_f L
        """
        per_sample_loss_grads = self._compute_per_sample_grad_loss(
            model, loss_fn, inputs, targets
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

    def set_track_running_stats(self, model, track=True):
        """
        Set the track_running_stats attribute of all BatchNorm layers to
        the specified value. This is a hack to use batch statistics
        without updating the buffers.
        """
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.track_running_stats = track
            self._track_running_stats = track
        return model


class SamplewiseCalculatorOpacus:
    def __init__(self):
        pass

    def compute(
        self,
        model,
        loss_fn,
        inputs,
        targets,
    ) -> Dict[str, torch.Tensor]:
        """
        Opacus backend is not implemented yet. Fail fast instead of
        returning None so callers can assume a dict is returned.
        """
        raise NotImplementedError(
            "SamplewiseCalculatorOpacus.compute is not implemented"
        )

# --- IGNORE ---
#
#    def __init2__(
#        self,
#        model: nn.Module,
#        optimizer: torch.optim.Optimizer,
#        criterion,
#        data_loader: DataLoader,
#        method: str,
#    ):
#        if method not in ["opacus", "functorch"]:
#            raise ValueError("method must be either 'opacus' or 'functorch'")
#        self.method = method
#
#        if self.method == "opacus":
#            self.privacy_engine = None
#            (
#                self.wrapped_model,
#                self.wrapped_optimizer,
#                self.wrapped_criterion,
#                self.wrapped_data_loader,
#            ) = self._wrap_model_for_privacy(
#                model=model,
#                optimizer=optimizer,
#                criterion=criterion,
#                data_loader=data_loader,
#            )
#
#    def _get_wrappings(self):
#        """
#        Function to retrieve model, optimizer, criterion,
#        and data loader wrappings
#        in the analyzer class.
#        """
#
#        return (
#            self.wrapped_model,
#            self.wrapped_optimizer,
#            self.wrapped_criterion,
#            self.wrapped_data_loader,
#        )
#
#    def _wrap_model_for_privacy(
#        self,
#        model: nn.Module,
#        optimizer: torch.optim.Optimizer,
#        criterion,
#        data_loader: DataLoader,
#    ) -> tuple:
#        """
#        Wrap model with opacus privacy engine. We just need the per sample
#        gradients so set max_grad_norm to 1e6 and noise_multiplier to 0 while
#        using 'ghost' mode.
#
#        Returns:
#            Tuple of (private_model, private_optimizer, private_criterion,
#                     private_data_loader)
#        """
# from opacus import PrivacyEngine
#        if self.method == "opacus":
#            if self.privacy_engine is None:
#                self.privacy_engine = PrivacyEngine()
#
#            (
#                private_model,
#                private_optimizer,
#                private_criterion,
#                private_data_loader,
#            ) = self.privacy_engine.make_private(
#                module=model,
#                optimizer=optimizer,
#                criterion=criterion,
#                data_loader=data_loader,
#                max_grad_norm=1e6,  # Effectively disable clipping
#                noise_multiplier=0,  # Effectively disable noise
#                grad_sample_mode="ghost",
#            )
#
#            return (
#                private_model,
#                private_optimizer,
#                private_criterion,
#                private_data_loader,
#            )
#
#    def _get_per_sample_gradient_norms_loss(
#        self, private_model
#    ) -> Optional[Any]:
#        """
#        Extract per-sample gradient norms from a privacy-wrapped model.
#
#        Args:
#            private_model: The model returned from wrap_model_for_privacy()
#
#        Returns:
#            Per-sample gradient norms tensor if available, None otherwise
#        """
#        if self.method == "opacus":
#            if hasattr(private_model, "per_sample_gradient_norms"):
#                return private_model.per_sample_gradient_norms
#
#        return None
#
#    def compute(self, data):
#        pass
#
