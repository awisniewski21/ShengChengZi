import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def set_seed(seed: int, gl_seed: int = 0):
    """ Set random seeds for reproducibility """
    if seed >= 0 and gl_seed >= 0:
        seed += gl_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

	# These may change convolution behavior (see https://pytorch.org/docs/stable/notes/randomness.html)
    if seed >= 0 and gl_seed >= 0: # Slower but more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else: # Faster but less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def set_device(obj, *args, **kwargs):
    """ Move torch to GPU/MPS if available """
    # Check for available accelerator devices
    has_accelerator = torch.cuda.is_available() # or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    
    if has_accelerator:
        if isinstance(obj, list):
            return [set_gpu(o, *args, **kwargs) for o in obj]
        elif isinstance(obj, dict):
            return {k: set_gpu(v, *args, **kwargs) for k, v in obj.items()}
        else:
            return set_gpu(obj, *args, **kwargs)
    return obj


###
### Helper Functions
###

def set_gpu(obj, distributed: bool = False, rank: int = 0):
    """ Move torch object to GPU/MPS or wrap with DDP """
    if obj is None:
        return None
    elif distributed and isinstance(obj, torch.nn.Module):
        if torch.cuda.is_available():
            return DDP(obj.cuda(), device_ids=[rank], output_device=rank, broadcast_buffers=True, find_unused_parameters=True)
        else:
            # For non-CUDA devices like MPS, we can't use device_ids
            return DDP(obj.to(get_device()), broadcast_buffers=True, find_unused_parameters=True)
    else:
        return obj.to(get_device())


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     return torch.device('mps')
    else:
        return torch.device('cpu')