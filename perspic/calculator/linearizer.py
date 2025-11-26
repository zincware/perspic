import copy
import io
from typing import Any, Callable, Optional, Tuple

import torch

from perspic.utils import set_track_running_stats


class Linearizer:
    """
    Class to perform multiple probe training steps on a model with different learning
    rates to get the linearization of the loss landscape.
    Args:
        eta_array: List of learning rates for probing. As this gets done on every
        training step of the analyzer, the eta_array is defined at initialization time.
    Returns:
        A dictionary mapping each learning rate to a tuple containing the original loss
        and the perturbed loss after the tiny step. If an error occurs during probing
        with a specific learning rate, the perturbed loss will be None for that entry.
    """

    def __init__(self, eta_array: list[float]):
        self.eta_array = eta_array

    def probe_train_step(
        self,
        model: torch.nn.Module,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        scheduler: Optional[Any] = None,
    ) -> dict[float, Tuple[float, Optional[float]]]:
        """
        Perform a tiny optimizer step (η) using batch-stats but zero-momentum,
        then undo everything. As this is done for multiple etas, we save/restore the model
        state only once before/after all etas have been probed.
        1. Save original model state (params + buffers) to in-memory buffer
        2. Compute and store loss and gradients on current step
        3. For each eta in eta_array:
            a. Propagate (stored) gradients on current batch param -= eta * grad
            b. Compute perturbed loss on current batch
            c. Undo tiny step: param += eta * grad (skip for last eta)
        3. Restore original model state
        4. Return dictionary of (original loss, perturbed loss) for each eta
        Args:
            model      : nn.Module
            criterion  : loss function
            x, y       : current batch input and targets
            scheduler  : optional lr-scheduler (snapshot & restore if provided)
        Returns:
            dict[float, Tuple[torch.Tensor, Optional[torch.Tensor]]]
        """
        # Save original state
        # 1a. Model (params + buffers) via in‐memory buffer to preserve device
        # placement
        orig_model_state = Linearizer._save_model_state(model)
        # 1b. Scheduler internals, if any
        orig_sched_state = copy.deepcopy(scheduler.state_dict()) if scheduler else None
        # 1c. Train/eval mode
        orig_mode = model.training

        device = (
            next(model.parameters()).device
            if any(p.requires_grad for p in model.parameters())
            else None
        )
        model = set_track_running_stats(
            model, track=False
        )  # Only use current batch stats
        model.train()  # still uses batch‐stats, but buffers won’t update
        results = {}

        try:
            loss = criterion(model(x), y)
            # prefer lightning’s manual backward if available
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
                    results[eta] = (
                        loss.detach().item(),
                        perturbed_loss.detach().item(),
                    )

                except Exception as e:
                    print(f"Error during probe with eta={eta}: {e}")
                    results[eta] = (loss.detach().item(), None)

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
            model = Linearizer._load_model_state(model, orig_model_state, device=device)
            if scheduler is not None:
                scheduler.load_state_dict(orig_sched_state)
            model = set_track_running_stats(model, track=True)
            model.train(orig_mode)

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
        state = torch.load(buf, map_location=map_loc)
        model.load_state_dict(state)
        return model
