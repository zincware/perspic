import torch.nn as nn


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
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = track
    return model
