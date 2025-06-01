from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import DDPMPipeline, DDPMScheduler, DiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel, UNet2DModel  # NOQA
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput  # NOQA
from diffusers.utils.torch_utils import randn_tensor
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from glyffuser import t5
from shengchengzi.scz_c2c_bi import Char2CharBiModel


class DiffusionPipeline_R2C_Glyff(DDPMPipeline):
    """
    Pipeline for image generation
    """

    def evaluate_texts_to_image_grid(self, *args, generator=None, seed: int = 0, **kwargs):
        gen_device = self.device if self.device.type not in ["cpu", "mps"] else "cpu"
        imgs = self(*args, generator=torch.Generator(device=gen_device).manual_seed(seed), **kwargs).images
        return np.tile((255 * imgs).clip(0, 255).astype(np.uint8), (1, 1, 1, 3))


class DiffusionPipeline_T2C_Glyff(DDPMPipeline):
    """
    Pipeline for text-to-image generation
    """
    unet: UNet2DConditionModel
    scheduler: DPMSolverMultistepScheduler

    @torch.no_grad()
    def __call__(
        self,
        texts: List[str],
        text_encoder: str = "google-t5/t5-small",
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size)
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            image = randn_tensor(image_shape, generator=generator, dtype=self.unet.dtype)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        self.scheduler.set_timesteps(num_inference_steps)

        # For text-to-image, compute embeddings and masks
        text_embeddings, masks = t5.t5_encode_text(texts, name=text_encoder, return_attn_mask=True, device=self.device)

        for t in self.progress_bar(self.scheduler.timesteps):
            model_output = self.unet(image, t, encoder_hidden_states=text_embeddings, encoder_attention_mask=masks).sample
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def evaluate_texts_to_image_grid(self, texts: List[str], text_encoder: str = "google-t5/t5-small", *args, generator=None, seed: int = 0, **kwargs):
        gen_device = self.device if self.device.type not in ["cpu", "mps"] else "cpu"
        imgs = self(texts, text_encoder=text_encoder, *args, generator=torch.Generator(device=gen_device).manual_seed(seed), **kwargs).images
        return np.tile((255 * imgs).clip(0, 255).astype(np.uint8), (1, 1, 1, 3))


class DiffusionPipeline_C2CBi_SCZ(DiffusionPipeline):
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


def evaluate_char_to_image_grid(model: UNet2DModel, val_dataloader: DataLoader, device: str):
    """
    Evaluate the Char2CharModel on the first batch of a given DataLoader and return the generated images as a grid.
    """
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for (b_imgs_src, b_imgs_trg, b_labels) in val_dataloader:
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

def evaluate_test_set(model, test_dataloader: DataLoader, device, noise_scheduler, task_name: str, writer=None, global_step=0):
    """
    Evaluate model on test set and return metrics.
    
    Args:
        model: The trained model
        test_dataloader: Test dataloader
        device: Device to run evaluation on
        noise_scheduler: Noise scheduler for diffusion models
        task_name: Type of task ("rand2char", "text2char", "char2char", etc.)
        writer: Optional tensorboard writer
        global_step: Global step for logging
        
    Returns:
        Dictionary of test metrics
    """
    if test_dataloader is None or len(test_dataloader) == 0:
        print("No test data available for evaluation.")
        return {}
    
    model.eval()
    test_loss = 0.0
    test_steps = 0
    
    print(f"Evaluating on test set ({len(test_dataloader)} batches)...")
    
    with torch.no_grad():
        if task_name == "rand2char":
            for test_imgs in test_dataloader:
                test_imgs = test_imgs.to(device)
                noise = torch.randn_like(test_imgs)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (test_imgs.shape[0],), device=device).long()
                test_imgs_noisy = noise_scheduler.add_noise(test_imgs, noise, timesteps)
                noise_pred = model(test_imgs_noisy, timesteps).sample
                test_loss += torch.nn.functional.mse_loss(noise_pred, noise).item()
                test_steps += 1
                
        elif task_name == "text2char":
            for test_imgs, test_texts_embed, test_masks in test_dataloader:
                test_imgs = test_imgs.to(device)
                test_texts_embed = test_texts_embed.to(device)
                test_masks = test_masks.to(device)
                
                noise = torch.randn_like(test_imgs)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (test_imgs.shape[0],), device=device).long()
                test_imgs_noisy = noise_scheduler.add_noise(test_imgs, noise, timesteps)
                noise_pred = model(test_imgs_noisy, timesteps, encoder_hidden_states=test_texts_embed).sample
                test_loss += torch.nn.functional.mse_loss(noise_pred, noise).item()
                test_steps += 1
                
        elif task_name in ["char2char", "char2char_bi"]:
            for test_imgs_src, test_imgs_trg, test_labels in test_dataloader:
                test_imgs_src = test_imgs_src.to(device)
                test_imgs_trg = test_imgs_trg.to(device)
                test_labels = test_labels.to(device) if test_labels is not None else None
                
                noise = torch.randn_like(test_imgs_trg)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (test_imgs_trg.shape[0],), device=device).long()
                test_imgs_noisy = noise_scheduler.add_noise(test_imgs_trg, noise, timesteps)
                noise_pred = model(test_imgs_noisy, timesteps).sample
                test_loss += torch.nn.functional.mse_loss(noise_pred, noise).item()
                test_steps += 1
    
    avg_test_loss = test_loss / test_steps if test_steps > 0 else 0.0
    
    test_metrics = {
        "test_loss": avg_test_loss,
        "test_samples": test_steps * test_dataloader.batch_size
    }
    
    print(f"Test Results - Loss: {avg_test_loss:.4f}, Samples: {test_metrics['test_samples']}")
    
    if writer is not None:
        writer.add_scalar("Loss/test", avg_test_loss, global_step)
        
    return test_metrics
