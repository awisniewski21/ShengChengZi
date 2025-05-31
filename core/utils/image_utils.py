import math
from typing import List

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch


def make_grid(images: List, rows: int, cols: int):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols*w, rows*h), color=(255, 255, 255))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def make_labeled_grid(images: List, prompt: str, steps: int, font_path: str | None = None, font_size: int = 20, margin: int = 10):
    assert len(images) == len(steps), "The number of images must match the number of steps"

    w, h = images[0].size
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    # Calculate the height of the grid including the margin for text
    total_height = h + margin + font_size
    total_width = w * len(images)
    grid_height = total_height + margin + font_size  # Add extra margin for the prompt
    grid = Image.new("RGB", size=(total_width, grid_height), color=(255, 255, 255))

    # Draw the text prompt at the top
    draw = ImageDraw.Draw(grid)
    prompt_text = f"Prompt: \"{prompt}\""
    prompt_width, prompt_height = draw.textbbox((0, 0), prompt_text, font=font)[2:4]
    prompt_x = (total_width - prompt_width) / 2
    prompt_y = margin / 2
    draw.text((prompt_x, prompt_y), prompt_text, fill="black", font=font)

    for i, (image, step) in enumerate(zip(images, steps)):
        # Calculate position to paste the image
        x = i * w
        y = margin + font_size

        # Paste the image
        grid.paste(image, box=(x, y))

        # Draw the step text
        step_text = f"Steps: {step}"
        text_width, text_height = draw.textbbox((0, 0), step_text, font=font)[2:4]
        text_x = x + (w - text_width) / 2
        text_y = y + h + margin / 2 - 8
        draw.text((text_x, text_y), step_text, fill="black", font=font)

    return grid

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