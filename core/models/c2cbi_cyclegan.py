from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from core.configs import TrainConfig_C2CBi_CycleGAN
from core.models import TrainModelBase
from core.utils.image_utils import compute_image_metrics, make_image_grid, to_out_img
from cyclegan_and_pix2pix.cyclegan_network import CycleGANNetwork, CycleGANOutput  # NOQA
from cyclegan_and_pix2pix.image_pool import ImagePool
from cyclegan_and_pix2pix.networks import GANLoss


class TrainModel_C2CBi_CycleGAN(TrainModelBase):
    """
    CycleGAN model for Bidirectional Character-to-Character (C2CBi) training
    """
    config: TrainConfig_C2CBi_CycleGAN
    net: CycleGANNetwork

    def __init__(
        self,
        *,
        optimizer_D: torch.optim.Optimizer,
        lr_scheduler_D: torch.optim.lr_scheduler.LRScheduler | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.optimizer_D = optimizer_D
        self.lr_scheduler_D = lr_scheduler_D

        self.fake_A_pool = ImagePool(self.config.pool_size)
        self.fake_B_pool = ImagePool(self.config.pool_size)

        self.loss_gan = GANLoss(self.config.gan_mode).to(self.device)
        self.loss_cycle = nn.L1Loss()
        self.loss_identity = nn.L1Loss()

    def train_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        real_A, real_B = batch_data
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)

        out = self.net.forward(real_A, real_B)

        # Update generators (with discriminators frozen)
        self.net.set_D_requires_grad(False)
        self.optimizer_G.zero_grad()
        loss_G = self.backward_G(out, real_A, real_B)
        loss_G.backward()
        self.optimizer_G.step()

        # Update discriminators
        self.net.set_D_requires_grad(True)
        self.optimizer_D.zero_grad()
        loss_D_A, loss_D_B = self.backward_D(out, real_A, real_B)
        loss_D_A.backward()
        loss_D_B.backward()
        self.optimizer_D.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss_G.item()

    def eval_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor], phase: str) -> Tuple[float, torch.Tensor, List[str] | None, Dict]:
        real_A, real_B = batch_data
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)

        out = self.net.forward(real_A, real_B)

        eval_loss = (self.loss_cycle(out.rec_A, real_A) + self.loss_cycle(out.rec_B, real_B)) / 2

        real_A_out = to_out_img(real_A, (0, 1))
        fake_B_out = to_out_img(out.fake_B, (0, 1))
        rec_A_out = to_out_img(out.rec_A, (0, 1))
        real_B_out = to_out_img(real_B, (0, 1))
        fake_A_out = to_out_img(out.fake_A, (0, 1))
        rec_B_out = to_out_img(out.rec_B, (0, 1))
        grid_img = make_image_grid([real_A_out, fake_B_out, rec_A_out, real_B_out, fake_A_out, rec_B_out])

        metrics = {"A": compute_image_metrics(out.rec_A, real_A), "B": compute_image_metrics(out.rec_B, real_B)}
        metrics = {k: (metrics["A"][k] + metrics["B"][k]) / 2 for k in metrics["A"].keys()}
        info = {"real_A": real_A, "fake_B": out.fake_B, "rec_A": out.rec_A, "real_B": real_B, "fake_A": out.fake_A, "rec_B": out.rec_B}

        return eval_loss.item(), grid_img, None, metrics, info

    def inference_step(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, List[str] | None, Dict]:
        real_AB = input_data.to(self.device)

        fake_B = self.net.netG_A(real_AB) # input -> B
        fake_A = self.net.netG_B(real_AB) # input -> A

        real_AB_out = to_out_img(real_AB, (0, 1))
        fake_B_out = to_out_img(fake_B, (-1, 1))
        fake_A_out = to_out_img(fake_A, (-1, 1))
        grid_img = make_image_grid([real_AB_out, fake_B_out, fake_A_out])

        return grid_img, None, {"real_AB": real_AB, "fake_B": fake_B, "fake_A": fake_A}

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

    def backward_G(self, out: CycleGANOutput, real_A: torch.Tensor, real_B: torch.Tensor) -> torch.Tensor:
        """ Calculate the loss for the generators """
        lambda_idt = self.config.lambda_identity
        lambda_A = self.config.lambda_A
        lambda_B = self.config.lambda_B

        # Identity losses
        if lambda_idt > 0:
            loss_idt_A = self.loss_identity(self.net.netG_A(real_B), real_B) * lambda_B * lambda_idt # G_A(B) should be close to B
            loss_idt_B = self.loss_identity(self.net.netG_B(real_A), real_A) * lambda_A * lambda_idt # G_B(A) should be close to A
        else:
            loss_idt_A = 0
            loss_idt_B = 0

        # GAN losses
        loss_G_A = self.loss_gan(self.net.netD_A(out.fake_B), True) # D_A(G_A(A))
        loss_G_B = self.loss_gan(self.net.netD_B(out.fake_A), True) # D_B(G_B(B))

        # Cycle losses
        loss_cycle_A = self.loss_cycle(out.rec_A, real_A) * lambda_A # G_B(G_A(A)) should be close to A
        loss_cycle_B = self.loss_cycle(out.rec_B, real_B) * lambda_B # G_A(G_B(B)) should be close to B

        # Combine losses
        return loss_idt_A + loss_idt_B + loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B

    def backward_D(self, out: CycleGANOutput, real_A: torch.Tensor, real_B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Calculate the loss for the discriminators """
        # Discriminator A losses
        fake_B = self.fake_B_pool.query(out.fake_B)
        loss_D_A_real = self.loss_gan(self.net.netD_A(real_B), True) # D_A(real_B) should be True
        loss_D_A_fake = self.loss_gan(self.net.netD_A(fake_B.detach()), False) # D_A(G_A(A)) should be False
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

        # Discriminator B losses
        fake_A = self.fake_A_pool.query(out.fake_A)
        loss_D_B_real = self.loss_gan(self.net.netD_B(real_A), True) # D_B(real_A) should be True
        loss_D_B_fake = self.loss_gan(self.net.netD_B(fake_A.detach()), False) # D_B(G_B(B)) should be False
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

        return loss_D_A, loss_D_B
