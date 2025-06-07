from pathlib import Path
from typing import List, Tuple

import einops
import torch
from PIL import ImageFont
from torchvision import transforms
from torchvision.utils import make_grid

from core.dataset.dataset_utils import create_image
from core.utils.repo_utils import get_repo_dir


def chars_to_image_tensor(input_chars: List[str], image_size: int, font_name: str, font_size: int | None) -> torch.Tensor:
    font_path = Path(get_repo_dir()) / "data" / "fonts" / f"{font_name}.ttf"
    font_size = font_size if font_size is not None else int(100 * (image_size / 128))
    font = ImageFont.truetype(font_path, font_size)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    return torch.stack([transform(create_image(c, image_size, font)) for c in input_chars])

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
