"""Small compatibility helpers for the offline, pre-installed environment."""

import inspect

import torch


def load_torch_file(path, map_location="cpu"):
    """Load trusted project data across old and new PyTorch releases.

    PyTorch releases that expose ``weights_only`` may default it differently.
    HAR files contain a trusted dict with tensors/arrays, so retain the historic
    full-object loading behavior without passing an unknown keyword to old builds.
    """

    kwargs = {"map_location": map_location}
    try:
        parameters = inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "weights_only" in parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def normalize_channels(x_data, eps=1e-6):
    """Normalize an ``[N, C, L]`` tensor channel-wise without torchvision."""

    mean = x_data.mean(dim=(0, 2), keepdim=True)
    std = x_data.std(dim=(0, 2), keepdim=True)
    return (x_data - mean) / std.clamp(min=eps)

