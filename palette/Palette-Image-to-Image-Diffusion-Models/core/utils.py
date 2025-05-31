import math
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid


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
    """ Move torch to GPU if available """
    if torch.cuda.is_available():
        if isinstance(obj, list):
            return [set_gpu(o, *args, **kwargs) for o in obj]
        elif isinstance(obj, dict):
            return {k: set_gpu(v, *args, **kwargs) for k, v in obj.items()}
        else:
            return set_gpu(obj, *args, **kwargs)
    return obj


def tensor_to_image(tensor: torch.Tensor, out_type=np.uint8, min_max=(-1, 1)):
    """
    Convert a torch Tensor into a NumPy image array.

    Args:
        tensor (torch.Tensor): 4D (B,C,H,W), 3D (C,H,W), or 2D (H,W) tensor, RGB channel order.
        out_type (type): Output type, default np.uint8.
        min_max (tuple): Clamp range, default (-1, 1).

    Returns:
        np.ndarray: 3D (H,W,C) or 2D (H,W) image, [0,255], np.uint8 (default).
    """
    tensor = tensor.clamp_(*min_max)
    n_dim = tensor.dim()

    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(f"Only support 4D, 3D and 2D tensor. But received with dimension: {n_dim}")

    if out_type == np.uint8:
        img_np = ((img_np + 1) * 127.5).round()
        # Note: numpy.uint8() does not round by default.

    return img_np.astype(out_type).squeeze()


###
### Helper Functions
###

def set_gpu(obj, distributed: bool = False, rank: int = 0):
    """ Move torch object to GPU or wrap with DDP """
    if obj is None:
        return None
    elif distributed and isinstance(obj, torch.nn.Module):
        return DDP(obj.cuda(), device_ids=[rank], output_device=rank, broadcast_buffers=True, find_unused_parameters=True)
    else:
        return obj.cuda()