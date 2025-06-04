from typing import List, Tuple

import einops
import numpy as np
import torch
from torchvision.utils import make_grid


def to_out_img(img: torch.Tensor, value_range: Tuple[int, int]) -> torch.Tensor:
    img = img.detach()
    if value_range == (-1, 1):
        img = ((img + 1) / 2).clamp(0, 1)
        value_range = (0, 1)
    if value_range == (0, 1):
        img = (img * 255)
        value_range = (0, 255)
    assert value_range == (0, 255), f"Unsupported value range: {value_range}"
    return img.to(torch.uint8).cpu()

def make_image_grid(imgs_out: List[torch.Tensor]) -> torch.Tensor:
    imgs_out_grid = torch.cat([i.unsqueeze(0) for i in imgs_out], dim=0)
    imgs_out_grid = einops.rearrange(imgs_out_grid, 'n b c h w -> (n b) c h w')
    return make_grid(imgs_out_grid, nrow=imgs_out[0].shape[0], padding=2)
