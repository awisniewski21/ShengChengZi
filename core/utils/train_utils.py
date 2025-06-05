import warnings

import torch


def get_device(verbose: bool = True) -> torch.device:
    """
    Get the appropriate device for training
    """
    if torch.cuda.is_available():
        if verbose: print(f"Using device: CUDA ({torch.cuda.get_device_name()})")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if verbose: print("Using device: MPS (Metal Performance Shaders)")
        return torch.device("mps")
    else:
        warnings.warn("CUDA and MPS are not available - defaulting to CPU")
        if verbose: print("Using device: CPU")
        return torch.device("cpu")
