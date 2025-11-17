from typing import List, Tuple

import torch.nn as nn


def set_track_running_states(model, track):
    """
    Recursively set track_running_stats for all BatchNorm layers in the model.
    Args:
        model : nn.Module
        track : bool, whether to track running stats or not
    Returns:
        model with updated BatchNorm layers
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = track
    return model


def save_bn_track_states(model) -> List[Tuple[nn.Module, bool]]:
    """
    Save the current track_running_stats state for all BatchNorm layers.
    Args:
        model : nn.Module
    Returns:
        List of (module, track_running_stats) tuples
    """
    return [
        (module, module.track_running_stats)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]


def restore_bn_track_states(bn_track_states) -> None:
    """
    Restore the track_running_stats state for BatchNorm layers.
    Args:
        bn_track_states : List of (module, track_running_stats) tuples
    """
    for module, track_state in bn_track_states:
        module.track_running_stats = track_state
