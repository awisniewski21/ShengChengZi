from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
from diffusers import DDPMScheduler, UNet2DModel

from configs import TrainConfig_C2CBi_SCZ
from core.models import TrainModelBase
from core.utils.image_utils import make_image_grid, to_out_img


class TrainModel_C2CBi_SCZ(TrainModelBase):
    """
    ShengChengZi model for Bidirectional Character-to-Character (C2CBi) training
    """
    config: TrainConfig_C2CBi_SCZ
    net: UNet2DModel

    def __init__(self, *, noise_scheduler: DDPMScheduler, **kwargs):
        super().__init__(task_prefix="train_c2cbi_shengchengzi", **kwargs)

        self.noise_scheduler = noise_scheduler

    def train_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        src_imgs, trg_imgs, labels = batch_data
        src_imgs = src_imgs.to(self.device)
        trg_imgs = trg_imgs.to(self.device)
        labels = labels.to(self.device)
        bs = src_imgs.shape[0]

        timesteps = torch.full((bs,), 1, device=self.device).long()

        pred_imgs = self.net(src_imgs, timesteps, class_labels=labels.unsqueeze(1)).sample

        loss = torch.nn.functional.mse_loss(pred_imgs.float(), trg_imgs.float(), reduction="mean")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item()

    def eval_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], phase: str, log_images: bool) -> float:
        src_imgs, trg_imgs, labels = batch_data
        src_imgs = src_imgs.to(self.device)
        trg_imgs = trg_imgs.to(self.device)
        labels = labels.to(self.device)
        bs = src_imgs.shape[0]

        timesteps = torch.full((bs,), 1, device=self.device).long()

        pred_imgs = self.net(src_imgs, timesteps, class_labels=labels.unsqueeze(1)).sample

        eval_loss = torch.nn.functional.mse_loss(pred_imgs.float(), trg_imgs.float(), reduction="mean")

        if log_images:
            src_imgs_out = to_out_img(src_imgs, (0, 1))
            trg_imgs_out = to_out_img(trg_imgs, (0, 1))
            pred_imgs_out = to_out_img(pred_imgs, (-1, 1))
            grid_img = make_image_grid([src_imgs_out, pred_imgs_out, trg_imgs_out])
            self.writer.add_image(f"{phase}/images", grid_img, self.current_epoch)
            grid_img_np = grid_img.permute(1, 2, 0).detach().cpu().numpy()
            plt.imshow(grid_img_np)
            plt.show()
            plt.imsave(self.images_dir / f"{phase}_epoch_{self.current_epoch}.png", grid_img_np)

        return eval_loss.item()

    def get_checkpoint_data(self) -> Dict:
        chkpt_data = super().get_checkpoint_data()
        chkpt_data["noise_scheduler_state"] = self.noise_scheduler.state_dict()
        return chkpt_data

    def load_checkpoint_data(self, chkpt_data: Dict):
        super().load_checkpoint_data(chkpt_data)
        self.noise_scheduler.load_state_dict(chkpt_data["noise_scheduler_state"])
