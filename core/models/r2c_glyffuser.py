from typing import Dict, List, Tuple

import torch
from diffusers import DDPMPipeline, DDPMScheduler, DPMSolverMultistepScheduler, UNet2DModel  # NOQA

from core.configs import TrainConfig_R2C_Glyff
from core.models import TrainModelBase
from core.utils.image_utils import make_image_grid, to_out_img


class TrainModel_R2C_Glyffuser(TrainModelBase):
    """
    Glyffuser model for Random-to-Character (R2C) training
    """
    config: TrainConfig_R2C_Glyff
    net: UNet2DModel

    def __init__(self, *, noise_scheduler: DDPMScheduler, inference_scheduler: DPMSolverMultistepScheduler, **kwargs):
        super().__init__(**kwargs)

        self.noise_scheduler = noise_scheduler
        self.inference_scheduler = inference_scheduler

    def train_step(self, batch_data: torch.Tensor) -> float:
        trg_imgs = batch_data.to(self.device)
        bs = trg_imgs.shape[0]

        noise = torch.randn_like(trg_imgs)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=self.device).long()
        trg_imgs_noisy = self.noise_scheduler.add_noise(trg_imgs, noise, timesteps)

        pred_noise = self.net(trg_imgs_noisy, timesteps, return_dict=False)[0]

        train_loss = torch.nn.functional.mse_loss(pred_noise, noise)

        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return train_loss.item()

    def eval_step(self, batch_data: torch.Tensor, phase: str) -> Tuple[float, torch.Tensor, List[str] | None]:
        eval_pipeline = DDPMPipeline(unet=self.net, scheduler=self.inference_scheduler)
        eval_pipeline.set_progress_bar_config(desc="Generating evaluation image grid...")

        pred_imgs = eval_pipeline(
            batch_size=self.config.eval_batch_size,
            generator=torch.Generator(device=self.device).manual_seed(self.config.seed),
            num_inference_steps=self.inference_scheduler.num_inference_steps,
            output_type="numpy",
        ).images

        pred_imgs_out = to_out_img(pred_imgs, (0, 1))
        grid_img = make_image_grid([pred_imgs_out])

        return 0, grid_img, None, {}, {"pred_imgs": pred_imgs}

    def inference_step(self, input_data: None = None) -> Tuple[torch.Tensor, List[str] | None, Dict]:
        inference_pipeline = DDPMPipeline(unet=self.net, scheduler=self.inference_scheduler)
        inference_pipeline.set_progress_bar_config(desc="Generating inference image grid...")

        pred_imgs = inference_pipeline(
            batch_size=self.config.eval_batch_size,
            generator=torch.Generator(device=self.device).manual_seed(self.config.seed),
            num_inference_steps=self.inference_scheduler.num_inference_steps,
            output_type="numpy",
        ).images

        pred_imgs_out = to_out_img(pred_imgs, (0, 1))
        grid_img = make_image_grid([pred_imgs_out])

        return grid_img, None, {"pred_imgs": pred_imgs}

    def get_checkpoint_data(self) -> Dict:
        chkpt_data = super().get_checkpoint_data()
        chkpt_data["noise_scheduler_state"] = dict(self.noise_scheduler.config)
        chkpt_data["inference_scheduler_state"] = dict(self.inference_scheduler.config)
        return chkpt_data

    def load_checkpoint_data(self, chkpt_data: Dict, phase: str):
        super().load_checkpoint_data(chkpt_data, phase)
        self.noise_scheduler = DDPMScheduler.from_config(chkpt_data["noise_scheduler_state"])
        self.inference_scheduler = DPMSolverMultistepScheduler.from_config(chkpt_data["inference_scheduler_state"])
