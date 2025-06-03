from typing import Dict

import matplotlib.pyplot as plt
import torch
from diffusers import DDPMPipeline, DDPMScheduler, DPMSolverMultistepScheduler, UNet2DModel  # NOQA

from configs import TrainConfig_R2C_Glyff
from core.models import TrainModelBase
from core.utils.image_utils import make_image_grid, to_out_img


class TrainModel_R2C_Glyffuser(TrainModelBase):
    """
    Glyffuser model for Random-to-Character (R2C) training
    """
    config: TrainConfig_R2C_Glyff
    net: UNet2DModel

    def __init__(self, *, noise_scheduler: DDPMScheduler, inference_scheduler: DPMSolverMultistepScheduler, **kwargs):
        super().__init__(task_prefix="train_r2c_glyffuser", **kwargs)

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

    def eval_step(self, batch_data: torch.Tensor, phase: str, log_images: bool) -> float:
        if log_images:
            eval_pipeline = DDPMPipeline(unet=self.net, scheduler=self.inference_scheduler)
            eval_pipeline.set_progress_bar_config(desc="Generating evaluation image grid...")

            pred_imgs = eval_pipeline(
                batch_size=self.config.eval_batch_size,
                generator=torch.Generator(device=self.device).manual_seed(self.config.seed),
                num_inference_steps=self.inference_scheduler.num_inference_steps,
                output_type="numpy",
            ).images

            pred_imgs_out = to_out_img(pred_imgs, (-1, 1))
            grid_img = make_image_grid([pred_imgs_out])
            self.writer.add_image(f"{phase}/images", grid_img, self.current_epoch)
            plt.imshow(grid_img.permute(1, 2, 0).detach().cpu().numpy())
            plt.show()

        return 0

    def get_checkpoint_data(self) -> Dict:
        chkpt_data = super().get_checkpoint_data()
        chkpt_data["noise_scheduler_state"] = self.noise_scheduler.state_dict()
        chkpt_data["inference_scheduler_state"] = self.inference_scheduler.state_dict()
        return chkpt_data

    def load_checkpoint_data(self, chkpt_data: Dict):
        super().load_checkpoint_data(chkpt_data)
        self.noise_scheduler.load_state_dict(chkpt_data["noise_scheduler_state"])
        self.inference_scheduler.load_state_dict(chkpt_data["inference_scheduler_state"])
