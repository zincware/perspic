import io
import copy
import torch
from typing import Tuple


class Linearizer:
    def __init__(self):
        pass

    def calculate_virtual_step(self, *args, **kwargs):
        # needs:
        # its own optimizer
        # zero-grad before step
        # step
        # scheduler
        # restore
        pass

    def probe_train_step(
        self,
        model,
        criterion,
        x,
        y,
        eta,
        scheduler=None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor
    ]:
        """
        Perform a tiny optimizer step (η) using batch-stats but zero-momentum,
        then undo everything.
        Args:
            model      : nn.Module
            criterion  : loss function
            x, y       : current batch input and targets
            eta        : small learning rate, e.g. 1e-5
            scheduler  : optional lr-scheduler (snapshot & restore if provided)
        Returns:
            logits, perturbed_logits, loss_value
        """
        # ————————————————————————————————
        # 1) Snapshot states
        # 1a. Model (params + buffers) via in‐memory buffer to preserve device
        # placement
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        orig_model_state = buf.getvalue()

        # 1b. Scheduler internals, if any
        orig_sched_state = None
        if scheduler is not None:
            orig_sched_state = copy.deepcopy(scheduler.state_dict())
        # 1c. Train/eval mode
        orig_mode = model.training
        model = set_track_running_stats(
            model, track=False
        )  # Only use current batch stats
        model.train()  # still uses batch‐stats, but buffers won’t update
        device = (
            next(model.parameters()).device
            if any(p.requires_grad for p in model.parameters())
            else None
        )

        try:
            # ————————————————————————————————
            # 2) Forward + backward
            loss = criterion(model(x), y)
            # prefer lightning’s manual backward if available
            if hasattr(model, "manual_backward"):
                model.manual_backward(loss)
            else:
                loss.backward()  # Compute gradients
            # ————————————————————————————————
            # Manually update the parameters
            for param in model.parameters():
                if param.grad is not None:
                    param.data -= eta * param.grad.data
            # Manually zero the gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.zero_()
            # ————————————————————————————————
            # 3) Probe perturbed network
            perturbed_loss = criterion(model(x), y)
        except Exception as e:
            # If an error occurs, we still want to restore the model state
            print(f"Error during probing step: {e}")
            perturbed_loss = None
            loss = None
        finally:
            # ————————————————————————————————
            # 4) Restore everything (even if an error occurred)
            # 4a. Model weights & buffers
            buf = io.BytesIO(orig_model_state)
            # ensure tensors map to the model device
            map_loc = device if device is not None else None
            state = torch.load(buf, map_location=map_loc)
            model.load_state_dict(state)
            # 4b. Scheduler state
            if scheduler is not None:
                scheduler.load_state_dict(orig_sched_state)
            # restore_bn_momentum(bn_layers)
            model = set_track_running_stats(model, track=True)
            # 4c. Restore train/eval mode
            if orig_mode:
                model.train()
            else:
                model.eval()
        return (loss, perturbed_loss)  # type: ignore


def set_track_running_stats(model, track=True):
    """
    Recursively set track_running_stats for all BatchNorm layers in the model.
    Args:
        model : nn.Module
        track : bool, whether to track running stats or not
    Returns:
        model with updated BatchNorm layers
    """
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = track
    return model
