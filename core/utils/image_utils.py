from pathlib import Path
from typing import Dict, List, Tuple

import einops
import numpy as np
import torch
from PIL import ImageFont
from skimage.metrics import structural_similarity as ssim
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

def to_out_img(img: torch.Tensor | np.ndarray, value_range: Tuple[int, int]) -> torch.Tensor:
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
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

def compute_image_metrics(pred: torch.Tensor, true: torch.Tensor) -> Dict:
    """
    Compute MSE, L1, SSIM, and PSNR between true and predicted images.
    """
    true = true.detach().cpu().numpy().squeeze()
    pred = pred.detach().cpu().numpy().squeeze()
    assert true.shape == pred.shape, f"Shape mismatch: {true.shape} vs {pred.shape}"

    image_metrics = {}

    image_metrics["mse"] = ((true - pred) ** 2).mean()
    image_metrics["l1"] = np.abs(true - pred).mean()
    image_metrics["ssim"] = ssim(true, pred, channel_axis=0, data_range=1.0)
    image_metrics["psnr"] = 20 * np.log10(1.0 / np.sqrt(image_metrics["mse"])) if image_metrics["mse"] > 0 else float("inf")

    return image_metrics