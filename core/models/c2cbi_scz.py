from typing import Dict, List, Tuple

import torch
from diffusers import DDPMScheduler, UNet2DModel

from core.configs import TrainConfig_C2CBi_SCZ
from core.models import TrainModelBase
from core.utils.image_utils import compute_image_metrics, make_image_grid, to_out_img  # NOQA


class TrainModel_C2CBi_SCZ(TrainModelBase):
    """
    ShengChengZi model for Bidirectional Character-to-Character (C2CBi) training
    """
    config: TrainConfig_C2CBi_SCZ
    net: UNet2DModel

    def __init__(self, *, noise_scheduler: DDPMScheduler, **kwargs):
        super().__init__(**kwargs)

        self.noise_scheduler = noise_scheduler

    def train_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        src_imgs, trg_imgs, labels = batch_data
        src_imgs = src_imgs.to(self.device)
        trg_imgs = trg_imgs.to(self.device)
        labels = labels.to(self.device)
        bs = src_imgs.shape[0]

        timesteps = torch.full((bs,), 1, device=self.device).long()

        pred_imgs = self.net(src_imgs, timesteps, class_labels=labels.unsqueeze(1)).sample
        pred_imgs = ((pred_imgs + 1) / 2).clamp(0, 1)

        loss = torch.nn.functional.mse_loss(pred_imgs, trg_imgs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item()

    def eval_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], phase: str) -> Tuple[float, torch.Tensor, List[str] | None, Dict]:
        src_imgs, trg_imgs, labels = batch_data
        src_imgs = src_imgs.to(self.device)
        trg_imgs = trg_imgs.to(self.device)
        labels = labels.to(self.device)
        bs = src_imgs.shape[0]

        timesteps = torch.full((bs,), 1, device=self.device).long()

        pred_imgs = self.net(src_imgs, timesteps, class_labels=labels.unsqueeze(1)).sample
        pred_imgs = ((pred_imgs + 1) / 2).clamp(0, 1)

        eval_loss = torch.nn.functional.mse_loss(pred_imgs, trg_imgs)

        src_imgs_out = to_out_img(src_imgs, (0, 1))
        trg_imgs_out = to_out_img(trg_imgs, (0, 1))
        pred_imgs_out = to_out_img(pred_imgs, (0, 1))
        grid_img = make_image_grid([src_imgs_out, pred_imgs_out, trg_imgs_out])

        metrics = compute_image_metrics(pred_imgs, trg_imgs)
        info = {"src_imgs": src_imgs, "pred_imgs": pred_imgs, "trg_imgs": trg_imgs, "labels": labels.tolist()}

        return eval_loss.item(), grid_img, None, metrics, info

    def inference_step(self, input_data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, List[str] | None, Dict]:
        src_imgs, labels = input_data
        src_imgs = src_imgs.to(self.device)
        labels = labels.to(self.device)
        bs = src_imgs.shape[0]

        timesteps = torch.full((bs,), 1, device=self.device).long()

        pred_imgs = self.net(src_imgs, timesteps, class_labels=labels.unsqueeze(1)).sample
        pred_imgs = ((pred_imgs + 1) / 2).clamp(0, 1)

        src_imgs_out = to_out_img(src_imgs, (0, 1))
        pred_imgs_out = to_out_img(pred_imgs, (0, 1))
        grid_img = make_image_grid([src_imgs_out, pred_imgs_out])

        return grid_img, None, {"src_imgs": src_imgs, "pred_imgs": pred_imgs, "labels": labels.tolist()}

    def get_checkpoint_data(self) -> Dict:
        chkpt_data = super().get_checkpoint_data()
        chkpt_data["noise_scheduler_config"] = dict(self.noise_scheduler.config)
        return chkpt_data

    def load_checkpoint_data(self, chkpt_data: Dict, phase: str):
        super().load_checkpoint_data(chkpt_data, phase)
        self.noise_scheduler = DDPMScheduler.from_config(chkpt_data["noise_scheduler_config"])
