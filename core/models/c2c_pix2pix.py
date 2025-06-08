from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from core.configs import TrainConfig_C2C_Pix2Pix
from core.models import TrainModelBase
from core.utils.image_utils import make_image_grid, to_out_img
from cyclegan_and_pix2pix.networks import GANLoss, define_D, define_G


class Pix2PixNetwork(nn.Module):
    """
    Pix2Pix network wrapper that contains all sub-networks
    """
    def __init__(self, config: TrainConfig_C2C_Pix2Pix):
        super().__init__()
        self.config = config
        
        # Define generator
        self.netG = define_G(
            config.input_nc, config.output_nc, config.ngf, 
            config.netG, config.norm, not config.no_dropout, 
            config.init_type, config.init_gain, []
        )
        
        # Define discriminator - takes both input and output images
        self.netD = define_D(
            config.input_nc + config.output_nc, config.ndf, 
            config.netD, config.n_layers_D, config.norm, 
            config.init_type, config.init_gain, []
        )
        
    def forward(self, x):
        # This is just a placeholder - Pix2Pix doesn't use standard forward pass
        return x


class TrainModel_C2C_Pix2Pix(TrainModelBase):
    """
    Pix2Pix model for Character-to-Character (C2C) training
    Adapted from pytorch-CycleGAN-and-pix2pix implementation
    """
    config: TrainConfig_C2C_Pix2Pix
    net: Pix2PixNetwork
    
    def __init__(self, *, optimizer_D: torch.optim.Optimizer, lr_scheduler_D=None, **kwargs):
        super().__init__(**kwargs)
        
        # Extract individual networks from the composite network
        self.netG = self.net.netG
        self.netD = self.net.netD
        
        # Store the discriminator optimizer and scheduler separately
        self.optimizer_D = optimizer_D
        self.lr_scheduler_D = lr_scheduler_D
        
        # Loss functions
        self.criterionGAN = GANLoss(self.config.gan_mode).to(self.device)
        self.criterionL1 = nn.L1Loss()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad for networks"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, real_A):
        """Run forward pass"""
        self.fake_B = self.netG(real_A)  # G(A)
        return self.fake_B

    def backward_D(self, real_A, real_B, fake_B):
        """Calculate GAN loss for the discriminator"""
        # Fake - stop backprop to generator by detaching fake_B
        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # Real
        real_AB = torch.cat((real_A, real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self, real_A, real_B, fake_B):
        """Calculate GAN and L1 loss for the generator"""
        # GAN loss - fool the discriminator
        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        # L1 loss - G(A) should be close to B
        self.loss_G_L1 = self.criterionL1(fake_B, real_B) * self.config.lambda_L1
        
        # Combined loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def train_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Training step for Pix2Pix"""
        src_imgs, trg_imgs = batch_data
        real_A = src_imgs.to(self.device)
        real_B = trg_imgs.to(self.device)
        
        # Forward pass
        fake_B = self.forward(real_A)
        
        # Update discriminator
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D(real_A, real_B, fake_B)
        self.optimizer_D.step()
        
        # Update generator
        self.set_requires_grad(self.netD, False)
        self.optimizer.zero_grad()  # This is the generator optimizer
        self.backward_G(real_A, real_B, fake_B)
        self.optimizer.step()
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        return self.loss_G.item()

    def eval_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor], phase: str) -> Tuple[float, torch.Tensor, List[str] | None]:
        """Evaluation step for Pix2Pix"""
        src_imgs, trg_imgs = batch_data
        real_A = src_imgs.to(self.device)
        real_B = trg_imgs.to(self.device)
        
        with torch.no_grad():
            fake_B = self.forward(real_A)
            # Calculate L1 loss for evaluation
            eval_loss = self.criterionL1(fake_B, real_B)

        # Create visualization grid
        real_A_out = to_out_img(real_A, (0, 1))
        fake_B_out = to_out_img(fake_B, (-1, 1))
        real_B_out = to_out_img(real_B, (0, 1))
        
        grid_img = make_image_grid([real_A_out, fake_B_out, real_B_out])

        return eval_loss.item(), grid_img, None

    def inference_step(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, List[str] | None]:
        """Inference step for Pix2Pix"""
        src_imgs = input_data.to(self.device)
        
        with torch.no_grad():
            fake_B = self.forward(src_imgs)
        
        src_imgs_out = to_out_img(src_imgs, (0, 1))
        fake_B_out = to_out_img(fake_B, (-1, 1))
        grid_img = make_image_grid([src_imgs_out, fake_B_out])

        return grid_img, None

    def get_checkpoint_data(self) -> Dict:
        """Get checkpoint data including all networks"""
        chkpt_data = super().get_checkpoint_data()
        chkpt_data.update({
            "netG_state_dict": self.netG.state_dict(),
            "netD_state_dict": self.netD.state_dict(),
            "optimizer_G_state_dict": self.optimizer.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
        })
        return chkpt_data

    def load_checkpoint_data(self, chkpt_data: Dict, phase: str):
        """Load checkpoint data for all networks"""
        super().load_checkpoint_data(chkpt_data, phase)
        
        self.netG.load_state_dict(chkpt_data["netG_state_dict"])
        self.netD.load_state_dict(chkpt_data["netD_state_dict"])
        
        if phase.lower() == "train" and self.optimizer is not None:
            self.optimizer.load_state_dict(chkpt_data["optimizer_G_state_dict"])
            self.optimizer_D.load_state_dict(chkpt_data["optimizer_D_state_dict"])
