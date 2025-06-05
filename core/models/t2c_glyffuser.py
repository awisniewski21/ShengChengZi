from typing import Dict, List, Tuple

import torch
from diffusers import DDPMPipeline, DDPMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel  # NOQA
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput  # NOQA
from diffusers.utils.torch_utils import randn_tensor

from configs import TrainConfig_T2C_Glyff
from core.models import TrainModelBase
from core.utils.image_utils import make_image_grid, to_out_img


class TrainModel_T2C_Glyffuser(TrainModelBase):
    """
    Glyffuser model for Text-to-Character (T2C) training
    """
    config: TrainConfig_T2C_Glyff
    net: UNet2DConditionModel

    def __init__(self, *, noise_scheduler: DDPMScheduler, inference_scheduler: DPMSolverMultistepScheduler, **kwargs):
        super().__init__(**kwargs)

        self.noise_scheduler = noise_scheduler
        self.inference_scheduler = inference_scheduler

    def train_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]) -> float:
        trg_imgs, src_texts_embed, _, _ = batch_data
        trg_imgs = trg_imgs.to(self.device)
        src_texts_embed = src_texts_embed.to(self.device)
        bs = trg_imgs.shape[0]

        noise = torch.randn_like(trg_imgs)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=self.device).long()
        trg_imgs_noisy = self.noise_scheduler.add_noise(trg_imgs, noise, timesteps)

        pred_noise = self.net(trg_imgs_noisy, timesteps, encoder_hidden_states=src_texts_embed).sample

        train_loss = torch.nn.functional.mse_loss(pred_noise, noise)

        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return train_loss.item()

    def eval_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]], phase: str) -> Tuple[float, torch.Tensor, List[str] | None]:
        eval_pipeline = DiffusionPipeline_T2C_Glyff(unet=self.net, scheduler=self.inference_scheduler)
        eval_pipeline.set_progress_bar_config(desc="Generating evaluation image grid...")

        trg_imgs, src_texts_embed, src_texts_mask, src_texts_raw = batch_data
        trg_imgs = trg_imgs.to(self.device)
        src_texts_embed = src_texts_embed.to(self.device)
        src_texts_mask = src_texts_mask.to(self.device)

        pred_imgs = eval_pipeline(
            src_texts_embed,
            src_texts_mask,
            batch_size=self.config.eval_batch_size,
            generator=torch.Generator(device=self.device).manual_seed(self.config.seed),
            num_inference_steps=self.inference_scheduler.num_inference_steps,
            output_type="numpy",
        ).images

        eval_loss = torch.nn.functional.mse_loss(pred_imgs, trg_imgs)

        trg_imgs_out = to_out_img(trg_imgs, (0, 1))
        pred_imgs_out = to_out_img(pred_imgs, (-1, 1))
        grid_img = make_image_grid([pred_imgs_out, trg_imgs_out])

        return eval_loss.item(), grid_img, src_texts_raw

    def get_checkpoint_data(self) -> Dict:
        chkpt_data = super().get_checkpoint_data()
        chkpt_data["noise_scheduler_state"] = self.noise_scheduler.state_dict()
        chkpt_data["inference_scheduler_state"] = self.inference_scheduler.state_dict()
        return chkpt_data

    def load_checkpoint_data(self, chkpt_data: Dict, phase: str):
        super().load_checkpoint_data(chkpt_data)
        self.noise_scheduler.load_state_dict(chkpt_data["noise_scheduler_state"])
        self.inference_scheduler.load_state_dict(chkpt_data["inference_scheduler_state"])


class DiffusionPipeline_T2C_Glyff(DDPMPipeline):
    """
    Inference diffusion pipeline for text-to-image generation
    """
    unet: UNet2DConditionModel
    scheduler: DPMSolverMultistepScheduler

    @torch.no_grad()
    def __call__(
        self,
        texts_embed: torch.Tensor,
        texts_mask: torch.Tensor,
        batch_size: int = 1,
        generator: torch.Generator | List[torch.Generator] | None = None,
        num_inference_steps: int = 1000,
        output_type: str | None = "pil",
        return_dict: bool = True,
    ) -> ImagePipelineOutput | Tuple:
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

        for t in self.progress_bar(self.scheduler.timesteps):
            model_output = self.unet(image, t, encoder_hidden_states=texts_embed, encoder_attention_mask=texts_mask).sample
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
