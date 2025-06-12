from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from core.configs import TrainConfig_C2C_Pix2Pix
from core.models import TrainModelBase
from core.utils.image_utils import compute_image_metrics, make_image_grid, to_out_img
from cyclegan_and_pix2pix.networks import GANLoss
from cyclegan_and_pix2pix.pix2pix_network import Pix2PixNetwork


class TrainModel_C2C_Pix2Pix(TrainModelBase):
    """
    Pix2Pix model for Character-to-Character (C2C) training
    """
    config: TrainConfig_C2C_Pix2Pix
    net: Pix2PixNetwork

    def __init__(self, *, optimizer_D: torch.optim.Optimizer, lr_scheduler_D=None, **kwargs):
        super().__init__(**kwargs)

        self.optimizer_D = optimizer_D
        self.lr_scheduler_D = lr_scheduler_D

        self.loss_gan = GANLoss(self.config.gan_mode).to(self.device)
        self.loss_l1 = nn.L1Loss()

    def train_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        real_A, real_B = batch_data
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)

        fake_B = self.net(real_A)

        # Update discriminator
        self.net.set_D_requires_grad(True)
        self.optimizer_D.zero_grad()
        loss_D = self.backward_D(fake_B, real_A, real_B)
        loss_D.backward()
        self.optimizer_D.step()

        # Update generator (with discriminator frozen)
        self.net.set_D_requires_grad(False)
        self.optimizer_G.zero_grad()
        loss_G = self.backward_G(fake_B, real_A, real_B)
        loss_G.backward()
        self.optimizer_G.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss_G.item()

    def eval_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor], phase: str) -> Tuple[float, torch.Tensor, List[str] | None, Dict, Dict]:
        real_A, real_B = batch_data
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)

        fake_B = self.net(real_A)

        eval_loss = self.loss_l1(fake_B, real_B)

        real_A_out = to_out_img(real_A, (0, 1))
        fake_B_out = to_out_img(fake_B, (0, 1))
        real_B_out = to_out_img(real_B, (0, 1))
        grid_img = make_image_grid([real_A_out, fake_B_out, real_B_out])

        metrics = compute_image_metrics(fake_B, real_B)
        info = {"real_A": real_A, "fake_B": fake_B, "real_B": real_B}

        return eval_loss.item(), grid_img, None, metrics, info

    def inference_step(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, List[str] | None, Dict]:
        real_A = input_data.to(self.device)

        fake_B = self.net(real_A)

        real_A_out = to_out_img(real_A, (0, 1))
        fake_B_out = to_out_img(fake_B, (0, 1))
        grid_img = make_image_grid([real_A_out, fake_B_out])

        return grid_img, None, {"real_A": real_A, "fake_B": fake_B}

    def get_checkpoint_data(self) -> Dict:
        chkpt_data = super().get_checkpoint_data()
        chkpt_data.update({
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
        })
        return chkpt_data

    def load_checkpoint_data(self, chkpt_data: Dict, phase: str):
        super().load_checkpoint_data(chkpt_data, phase)

        if phase.lower() == "train":
            self.optimizer_D.load_state_dict(chkpt_data["optimizer_D_state_dict"])

    @property
    def optimizer_G(self) -> torch.optim.Optimizer | None:
        return self.optimizer

    @property
    def lr_scheduler_G(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        return self.lr_scheduler


    ###
    ### Helper Functions
    ###

    def backward_G(self, fake_B: torch.Tensor, real_A: torch.Tensor, real_B: torch.Tensor) -> torch.Tensor:
        """ Calculate the loss for the generator """
        # GAN loss (fool discriminator)
        pred_fake = self.net.netD(torch.cat((real_A, fake_B), 1))
        loss_G_GAN = self.loss_gan(pred_fake, True) # D(G(A) | A) should be True

        # L1 loss
        loss_G_L1 = self.loss_l1(fake_B, real_B) * self.config.lambda_L1 # G(A) should be close to B

        # Combine losses
        return loss_G_GAN + loss_G_L1

    def backward_D(self, fake_B: torch.Tensor, real_A: torch.Tensor, real_B: torch.Tensor) -> torch.Tensor:
        """ Calculate the loss for the discriminator """
        # Fake GAN loss
        pred_fake = self.net.netD(torch.cat((real_A, fake_B), 1).detach())
        loss_D_fake = self.loss_gan(pred_fake, False) # D(G(A) | A) should be False

        # Real GAN loss
        pred_real = self.net.netD(torch.cat((real_A, real_B), 1))
        loss_D_real = self.loss_gan(pred_real, True) # D(B | A) should be True

        # Combine losses
        return (loss_D_fake + loss_D_real) * 0.5
