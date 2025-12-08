import copy
import io
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple

import torch


class BaseLinearizer(ABC):
    """
    Abstract base class for linearization methods.

    Linearizers compute the linear response of the loss landscape, which can be used
    to analyze training dynamics and compute coupling values.

    Subclasses must implement:
        - exact_linearizer: Property indicating if this is an exact method
        - compute: Method to compute the linear response

    Returns:
        A dictionary mapping each learning rate (eta) to a tuple containing:
        (original_loss, perturbed_loss, delta_loss)
        where delta_loss = perturbed_loss - original_loss
    """

    @property
    @abstractmethod
    def exact_linearizer(self) -> bool:
        """Return True if this linearizer uses exact gradient norm computation."""
        pass

    @abstractmethod
    def compute(
        self,
        model: torch.nn.Module,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        scheduler: Optional[Any] = None,
    ) -> dict[float, Tuple[float, Optional[float], Optional[float]]]:
        """
        Compute the linear response of the loss landscape.

        Args:
            model: nn.Module
            criterion: loss function
            x, y: current batch input and targets
            scheduler: optional lr-scheduler (snapshot & restore if provided)

        Returns:
            dict[float, Tuple[float, Optional[float], Optional[float]]]:
                Dictionary mapping eta to (loss, perturbed_loss, delta_loss)
        """
        pass


class ApproximateLinearizer(BaseLinearizer):
    """
    Linearizer that approximates the linear response using virtual gradient steps.

    Performs multiple probe training steps on a model with different learning
    rates to approximate the linearization of the loss landscape via first-order
    Taylor expansion: L(θ - η∇L) ≈ L(θ) - η||∇L||²

    Args:
        eta_array: List of learning rates for probing. Required parameter.

    Returns:
        A dictionary mapping each learning rate to a tuple containing:
        (original_loss, perturbed_loss, delta_loss)
        If an error occurs during probing with a specific learning rate,
        the perturbed_loss and delta_loss will be None for that entry.
    """

    def __init__(self, eta_array: list[float]):
        if eta_array is None or len(eta_array) == 0:
            raise ValueError(
                "eta_array is required for ApproximateLinearizer and must be non-empty"
            )
        self.eta_array = eta_array

    @property
    def exact_linearizer(self) -> bool:
        """Return False as this uses approximate virtual step method."""
        return False

    def compute(
        self,
        model: torch.nn.Module,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        scheduler: Optional[Any] = None,
    ) -> dict[float, Tuple[float, Optional[float], Optional[float]]]:
        """
        Perform a tiny optimizer step (η) using batch-stats but zero-momentum,
        then undo everything. As this is done for multiple etas, we save/restore
        the model state only once before/after all etas have been probed.

        1. Save original model state (params + buffers) to in-memory buffer
        2. Compute and store loss and gradients on current step
        3. For each eta in eta_array:
            a. Propagate (stored) gradients on current batch param -= eta * grad
            b. Compute perturbed loss on current batch
            c. Undo tiny step: param += eta * grad (skip for last eta)
        4. Restore original model state
        5. Return dictionary of (original loss, perturbed loss, delta loss) for each eta

        Args:
            model: nn.Module
            criterion: loss function
            x, y: current batch input and targets
            scheduler: optional lr-scheduler (snapshot & restore if provided)

        Returns:
            dict[float, Tuple[float, Optional[float], Optional[float]]]
        """
        # Save original state
        # 1a. Model (params + buffers) via in-memory buffer to preserve device
        # placement
        orig_model_state = ApproximateLinearizer._save_model_state(model)
        # 1b. Scheduler internals, if any
        orig_sched_state = copy.deepcopy(scheduler.state_dict()) if scheduler else None

        device = (
            next(model.parameters()).device
            if any(p.requires_grad for p in model.parameters())
            else None
        )

        results = {}
        try:
            loss = criterion(model(x), y)
            # prefer lightning's manual backward if available
            if hasattr(model, "manual_backward"):
                model.manual_backward(loss)  # type: ignore
            else:
                loss.backward()
            param_grads = [
                (param, param.grad.clone() if param.grad is not None else None)
                for param in model.parameters()
            ]
            # probe each eta
            for i, eta in enumerate(self.eta_array):
                try:
                    with torch.no_grad():
                        for param, grad in param_grads:
                            if grad is not None:
                                param.data.sub_(grad, alpha=eta)

                    perturbed_loss = criterion(model(x), y)
                    loss_val = loss.detach().item()
                    perturbed_loss_val = perturbed_loss.detach().item()
                    results[eta] = (
                        loss_val,
                        perturbed_loss_val,
                        perturbed_loss_val - loss_val,
                    )

                except Exception as e:
                    print(f"Error during probe with eta={eta}: {e}")
                    results[eta] = (loss.detach().item(), None, None)

                finally:
                    if i < len(self.eta_array) - 1:
                        with torch.no_grad():
                            for param, grad in param_grads:
                                if grad is not None:
                                    param.data.add_(grad, alpha=eta)
            # Manually zero the gradients
            model.zero_grad()
            # lower memory footprint -> see torch/optim/optimizer.py#L997

        except Exception as e:
            # TODO: add logger warning here (when logger is available)
            # If an error occurs, we still want to restore the model state
            print(f"Error during probing step: {e}")
        finally:
            # Restore everything (even if an error occurred)
            model = ApproximateLinearizer._load_model_state(
                model, orig_model_state, device=device
            )
            if scheduler is not None:
                scheduler.load_state_dict(orig_sched_state)

        return results

    @staticmethod
    def _save_model_state(model: torch.nn.Module) -> bytes:
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return buf.getvalue()

    @staticmethod
    def _load_model_state(
        model: torch.nn.Module,
        state_bytes: bytes,
        device: Optional[torch.device] = None,
    ) -> torch.nn.Module:
        buf = io.BytesIO(state_bytes)
        # ensure tensors map to the model device
        map_loc = device if device is not None else None
        state = torch.load(buf, map_location=map_loc, weights_only=True)
        model.load_state_dict(state)

        return model


class ExactLinearizer(BaseLinearizer):
    """
    Linearizer that computes the exact linear response using ||∇L||².

    This method directly computes the gradient norm squared, which is the
    exact first-order term in the Taylor expansion of the loss landscape.
    This is cheaper and more accurate than the virtual step approximation.

    The linear response is: ΔL = -||∇L||²

    Args:
        eta_array: Not supported for ExactLinearizer. Will raise an error if provided.

    Returns:
        A dictionary with key -1 mapping to a tuple containing:
        (loss, loss - ||∇L||², -||∇L||²)
        The eta=-1 convention indicates this is the exact method.
    """

    def __init__(self, eta_array: Optional[list[float]] = None):
        if eta_array is not None:
            raise ValueError(
                "eta_array is not supported for ExactLinearizer. "
                "Use ApproximateLinearizer if you need to probe multiple learning rates."
            )

    @property
    def exact_linearizer(self) -> bool:
        """Return True as this uses exact gradient norm computation."""
        return True

    def compute(
        self,
        model: torch.nn.Module,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        scheduler: Optional[Any] = None,
    ) -> dict[float, Tuple[float, Optional[float], Optional[float]]]:
        """
        Compute the exact linear response using ||∇L||².

        This method computes the gradient norm squared directly, which gives
        the exact first-order Taylor expansion term without approximation.

        Args:
            model: nn.Module
            criterion: loss function
            x, y: current batch input and targets
            scheduler: not used, included for API compatibility

        Returns:
            dict[float, Tuple[float, float, float]]:
                Dictionary with key -1 mapping to (loss, loss - ||∇L||², -||∇L||²)
        """
        model.zero_grad()

        loss = criterion(model(x), y)
        # prefer lightning's manual backward if available
        if hasattr(model, "manual_backward"):
            model.manual_backward(loss)  # type: ignore
        else:
            loss.backward()

        # Compute ||∇L||²
        grad_norm_squared = sum(
            (p.grad**2).sum().item() for p in model.parameters() if p.grad is not None
        )

        model.zero_grad()

        loss_val = loss.detach().item()
        delta_loss = -grad_norm_squared
        perturbed_loss = loss_val + delta_loss  # loss - ||∇L||²

        # Return with eta=-1 to indicate exact method
        return {-1: (loss_val, perturbed_loss, delta_loss)}


# Backwards compatibility alias
Linearizer = ApproximateLinearizer
