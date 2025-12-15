from typing import Callable, Optional, Tuple

import torch


class Linearizer:
    """
    Computes the exact linear response of the loss landscape using gradient dot
    products.

    This linearizer computes:
    - "self" response: ||∇L(x1, y1)||² — gradient norm squared on the training batch
    - "cross" response: ∇L(x1, y1)^T ∇L(x2, y2) — dot product between gradients
      from two different batches (optional)

    The linear response represents the first-order Taylor expansion term:
    ΔL ≈ -η * (gradient dot product)

    For "self": ΔL = -||∇L||²
    For "cross": ΔL = -∇L₁^T ∇L₂

    Returns:
        A dictionary with keys "self" and optionally "cross", each mapping to:
        (loss, perturbed_loss, delta_loss)
        where delta_loss is the negative gradient dot product.
    """

    def __init__(self):
        pass

    def compute(
        self,
        model: torch.nn.Module,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x1: torch.Tensor,
        y1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        y2: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[Tuple[float, float, float]]]:
        """
        Compute the linear response using gradient dot products.

        Computes the "self" response (||∇L₁||²) and optionally the "cross" response
        (∇L₁^T ∇L₂) when a second batch is provided.

        Args:
            model: nn.Module
            criterion: loss function
            x1, y1: primary batch input and targets (training batch)
            x2, y2: optional secondary batch for cross-response computation

        Returns:
            dict[str, Optional[Tuple[float, float, float]]]:
                {
                    "self": (loss1, perturbed_loss1, delta_loss1),
                    "cross": (loss2, perturbed_loss2, delta_loss2) or None
                }
                where delta_loss is the negative gradient dot product.
        """
        model.zero_grad()

        # Compute loss and gradients on batch 1
        loss1 = criterion(model(x1), y1)
        if hasattr(model, "manual_backward"):
            model.manual_backward(loss1)
        else:
            loss1.backward()

        # Only store gradients if we need them for cross-response
        # This avoids doubling memory usage when only computing "self"
        if x2 is not None and y2 is not None:
            grads1 = [
                p.grad.clone() if p.grad is not None else None
                for p in model.parameters()
            ]

        # Compute ||∇L₁||² (self response)
        grad_norm_squared = sum(
            (p.grad**2).sum().item() for p in model.parameters() if p.grad is not None
        )

        loss1_val = loss1.detach().item()
        delta_loss_self = -grad_norm_squared
        perturbed_loss_self = loss1_val + delta_loss_self

        results = {
            "self": (loss1_val, perturbed_loss_self, delta_loss_self),
            "cross": None,
        }

        # Compute cross response if second batch is provided
        if x2 is not None and y2 is not None:
            model.zero_grad()

            # Compute loss and gradients on batch 2
            loss2 = criterion(model(x2), y2)
            if hasattr(model, "manual_backward"):
                model.manual_backward(loss2)
            else:
                loss2.backward()

            # Compute ∇L₁^T ∇L₂ (cross response)
            cross_dot_product = sum(
                (g1 * p.grad).sum().item()
                for g1, p in zip(grads1, model.parameters())
                if g1 is not None and p.grad is not None
            )

            loss2_val = loss2.detach().item()
            delta_loss_cross = -cross_dot_product
            perturbed_loss_cross = loss2_val + delta_loss_cross

            results["cross"] = (loss2_val, perturbed_loss_cross, delta_loss_cross)

        model.zero_grad()

        return results
