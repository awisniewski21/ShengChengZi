import copy
from typing import Dict, Tuple

import torch

from configs import TrainConfig_C2C_Palette
from core.models import TrainModelBase
from core.utils.image_utils import make_image_grid, to_out_img
from palette.models.palette_network import PaletteNetwork
from palette.models.utils import update_model_average


class TrainModel_C2C_Palette(TrainModelBase):
    """
    Palette model for Character-to-Character (C2C) training
    """
    config: TrainConfig_C2C_Palette
    net: PaletteNetwork

    def __init__(self, **kwargs):
        super().__init__(task_prefix="train_c2c_palette", **kwargs)

        self.net.set_new_noise_schedule(self.device)

        if self.config.ema_enabled:
            self.net_ema = copy.deepcopy(self.net)
            self.net_ema = self.net_ema.to(self.device)

    def train_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        src_imgs, trg_imgs = batch_data
        src_imgs = src_imgs.to(self.device)
        trg_imgs = trg_imgs.to(self.device)

        train_loss = self.net(trg_imgs, y_cond=src_imgs)

        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()

        if self.config.ema_enabled:
            if self.global_step > self.config.ema_start and self.global_step % self.config.ema_iter == 0:
                update_model_average(self.net, self.net_ema, self.config.ema_decay)

        return train_loss.item()

    def eval_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor], phase: str, log_images: bool) -> float:
        src_imgs, trg_imgs = batch_data
        src_imgs = src_imgs.to(self.device)
        trg_imgs = trg_imgs.to(self.device)

        pred_imgs, _ = self.net.restoration(src_imgs, sample_num=self.config.sample_num)

        eval_loss = torch.nn.functional.mse_loss(pred_imgs, trg_imgs)

        if log_images:
            src_imgs_out = to_out_img(src_imgs, (0, 1))
            trg_imgs_out = to_out_img(trg_imgs, (0, 1))
            pred_imgs_out = to_out_img(pred_imgs, (-1, 1))
            grid_img = make_image_grid([src_imgs_out, pred_imgs_out, trg_imgs_out])
            self.writer.add_image(f"{phase}/images", grid_img, self.current_epoch)

        return eval_loss.item()

    def get_checkpoint_data(self) -> Dict:
        chkpt_data = super().get_checkpoint_data()
        if self.config.ema_enabled:
            chkpt_data["net_ema_state_dict"] = self.net_ema.state_dict()
        return chkpt_data

    def load_checkpoint_data(self, chkpt_data: Dict):
        super().load_checkpoint_data(chkpt_data)
        if self.config.ema_enabled:
            self.net_ema.load_state_dict(chkpt_data["net_ema_state_dict"])
