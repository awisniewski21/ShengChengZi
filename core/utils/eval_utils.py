from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline, UNet2DModel, DDPMScheduler, DPMSolverMultistepScheduler  # NOQA
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput  # NOQA
from diffusers.utils.torch_utils import randn_tensor
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from shengchengzi.models.scz_c2c_bi import Char2CharBiModel


class DiffusionPipelineChar2CharBi(DiffusionPipeline):
    """
    Pipeline for text-to-image generation
    """
    model: Char2CharBiModel
    scheduler: DPMSolverMultistepScheduler

    def __init__(self, model: Char2CharBiModel, scheduler: DDPMScheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        src_imgs: torch.Tensor,
        trg_labels: torch.Tensor,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        device = self.model.vae.device
        src_imgs = src_imgs.to(device)
        trg_labels = trg_labels.to(device)

        # Encode source images to latent and reshape for encoder_hidden_states
        src_latent = self.model.vae.encode(src_imgs).latent_dist.sample() * self.model.vae.config.scaling_factor
        encoder_hidden_states = src_latent.squeeze(1)

        # Start from random noise in latent space
        latent_shape = src_latent.shape
        if device.type == "mps":
            latents = randn_tensor(latent_shape, generator=generator, dtype=src_latent.dtype)
            latents = latents.to(device)
        else:
            latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=src_latent.dtype)

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            unet_out = self.model.unet(latents, t, encoder_hidden_states, class_labels=trg_labels.unsqueeze(1)).sample
            latents = self.scheduler.step(unet_out, t, latents, generator=generator).prev_sample

        # self.model.vae.decoder.incoming_skip_acts = self.model.vae.encoder.current_down_blocks

        imgs = self.model.vae.decode(latents / self.model.vae.config.scaling_factor).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            imgs = self.numpy_to_pil(imgs)

        if not return_dict:
            return (imgs,)

        return ImagePipelineOutput(images=imgs)

    def evaluate_char_to_image_grid(self, src_imgs: torch.Tensor, trg_labels: torch.Tensor, *args, generator=None, seed: int = 0, **kwargs):
        gen_device = self.model.vae.device if self.model.vae.device.type not in ["cpu", "mps"] else "cpu"
        imgs = self(src_imgs, trg_labels, *args, generator=torch.Generator(device=gen_device).manual_seed(seed), **kwargs).images
        return np.tile((255 * imgs).clip(0, 255).astype(np.uint8), (1, 1, 1, 3))


def evalaute_char_to_image_grid(model: UNet2DModel, eval_dataloader: DataLoader, device: str):
    """
    Evaluate the Char2CharModel on the first batch of a given DataLoader and return the generated images as a grid.
    """
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for (b_imgs_src, b_imgs_trg, b_labels) in eval_dataloader:
            b_imgs_src = b_imgs_src.to(device)
            b_labels = b_labels.to(device)
            B, C, H, W = b_imgs_src.shape

            timesteps = torch.full((b_imgs_src.shape[0],), 1, device=device).long()
            pred_imgs = model(b_imgs_src, timesteps, class_labels=b_labels.unsqueeze(1)).sample

            # Convert pred_imgs from (B, 1, H, W) [-1, 1] to (B, 1, H, W) [0, 255]
            pred_imgs = (pred_imgs / 2 + 0.5).clamp(0, 1)
            pred_imgs = (pred_imgs * 255).to(torch.uint8).cpu()

            b_imgs_src = (b_imgs_src * 255).to(torch.uint8).cpu()
            b_imgs_trg = (b_imgs_trg * 255).to(torch.uint8).cpu()

            triplets = torch.stack([b_imgs_src, pred_imgs, b_imgs_trg]).transpose(1, 0).reshape(B*3, C, H, W)
            return make_grid(triplets, nrow=3, padding=2, normalize=False)
