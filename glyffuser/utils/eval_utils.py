from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import DDPMPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

from glyffuser.models import t5


class DiffusionPipelineRand2Char(DDPMPipeline):
    """
    Pipeline for image generation
    """

    def evaluate_texts_to_image_grid(self, *args, generator=None, seed: int = 0, **kwargs):
        gen_device = self.device if self.device.type not in ["cpu", "mps"] else "cpu"
        imgs = self(*args, generator=torch.Generator(device=gen_device).manual_seed(seed), **kwargs).images
        return np.tile((255 * imgs).clip(0, 255).astype(np.uint8), (1, 1, 1, 3))


class DiffusionPipelineText2Char(DDPMPipeline):
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
