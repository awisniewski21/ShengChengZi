from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from configs import TrainConfig_C2C_CycleGAN
from core.models import TrainModelBase
from core.utils.image_utils import make_image_grid, to_out_img
from cyclegan_and_pix2pix.image_pool import ImagePool
from cyclegan_and_pix2pix.networks import GANLoss, define_D, define_G


class CycleGANNetwork(nn.Module):
    """
    CycleGAN network wrapper that contains all sub-networks
    """
    def __init__(self, config: TrainConfig_C2C_CycleGAN):
        super().__init__()
        self.config = config
        
        # Define generators
        self.netG_A = define_G(
            config.input_nc, config.output_nc, config.ngf, 
            config.netG, config.norm, not config.no_dropout, 
            config.init_type, config.init_gain, []
        )
        self.netG_B = define_G(
            config.output_nc, config.input_nc, config.ngf, 
            config.netG, config.norm, not config.no_dropout, 
            config.init_type, config.init_gain, []
        )
        
        # Define discriminators
        self.netD_A = define_D(
            config.output_nc, config.ndf, config.netD,
            config.n_layers_D, config.norm, config.init_type, 
            config.init_gain, []
        )
        self.netD_B = define_D(
            config.input_nc, config.ndf, config.netD,
            config.n_layers_D, config.norm, config.init_type, 
            config.init_gain, []
        )
        
    def forward(self, x):
        # This is just a placeholder - CycleGAN doesn't use standard forward pass
        return x


class TrainModel_C2C_CycleGAN(TrainModelBase):
    """
    CycleGAN model for Character-to-Character (C2C) training
    Adapted from pytorch-CycleGAN-and-pix2pix implementation
    """
    config: TrainConfig_C2C_CycleGAN
    net: CycleGANNetwork
    
    def __init__(self, *, optimizer_D: torch.optim.Optimizer, lr_scheduler_D=None, **kwargs):
        super().__init__(**kwargs)
        
        # Extract individual networks from the composite network
        self.netG_A = self.net.netG_A
        self.netG_B = self.net.netG_B
        self.netD_A = self.net.netD_A
        self.netD_B = self.net.netD_B
        
        # Store the discriminator optimizer and scheduler separately
        self.optimizer_D = optimizer_D
        self.lr_scheduler_D = lr_scheduler_D
        
        # Image pools for storing previously generated images
        self.fake_A_pool = ImagePool(self.config.pool_size)
        self.fake_B_pool = ImagePool(self.config.pool_size)
        
        # Loss functions
        self.criterionGAN = GANLoss(self.config.gan_mode).to(self.device)
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, real_A, real_B):
        """Run forward pass"""
        self.fake_B = self.netG_A(real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self, real_B):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, real_B, fake_B)

    def backward_D_B(self, real_A):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, real_A, fake_A)

    def backward_G(self, real_A, real_B):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.config.lambda_identity
        lambda_A = self.config.lambda_A
        lambda_B = self.config.lambda_B
        
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed
            self.idt_A = self.netG_A(real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed
            self.idt_B = self.netG_B(real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, real_B) * lambda_B
        # Combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def train_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Training step for CycleGAN"""
        src_imgs, trg_imgs = batch_data
        real_A = src_imgs.to(self.device)
        real_B = trg_imgs.to(self.device)
        
        # Forward pass
        self.forward(real_A, real_B)
        
        # Update generators G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer.zero_grad()  # This is the generator optimizer
        self.backward_G(real_A, real_B)
        self.optimizer.step()
        
        # Update discriminators D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A(real_B)
        self.backward_D_B(real_A)
        self.optimizer_D.step()
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # Return total generator loss
        return self.loss_G.item()

    def eval_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor], phase: str) -> Tuple[float, torch.Tensor, List[str] | None]:
        """Evaluation step for CycleGAN"""
        src_imgs, trg_imgs = batch_data
        real_A = src_imgs.to(self.device)
        real_B = trg_imgs.to(self.device)
        
        with torch.no_grad():
            self.forward(real_A, real_B)
            # Calculate cycle consistency loss for evaluation
            eval_loss = (self.criterionCycle(self.rec_A, real_A) + 
                        self.criterionCycle(self.rec_B, real_B)) / 2

        # Create visualization grid
        real_A_out = to_out_img(real_A, (0, 1))
        fake_B_out = to_out_img(self.fake_B, (-1, 1))
        rec_A_out = to_out_img(self.rec_A, (-1, 1))
        real_B_out = to_out_img(real_B, (0, 1))
        fake_A_out = to_out_img(self.fake_A, (-1, 1))
        rec_B_out = to_out_img(self.rec_B, (-1, 1))
        
        grid_img = make_image_grid([
            real_A_out, fake_B_out, rec_A_out,
            real_B_out, fake_A_out, rec_B_out
        ])

        return eval_loss.item(), grid_img, None

    def inference_step(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, List[str] | None]:
        """Inference step for CycleGAN"""
        src_imgs = input_data.to(self.device)
        
        with torch.no_grad():
            fake_B = self.netG_A(src_imgs)  # A -> B translation
            fake_A = self.netG_B(src_imgs)  # Can also do B -> A if needed
        
        src_imgs_out = to_out_img(src_imgs, (0, 1))
        fake_B_out = to_out_img(fake_B, (-1, 1))
        grid_img = make_image_grid([src_imgs_out, fake_B_out])

        return grid_img, None

    def get_checkpoint_data(self) -> Dict:
        """Get checkpoint data including all networks"""
        chkpt_data = super().get_checkpoint_data()
        chkpt_data.update({
            "netG_A_state_dict": self.netG_A.state_dict(),
            "netG_B_state_dict": self.netG_B.state_dict(),
            "netD_A_state_dict": self.netD_A.state_dict(),
            "netD_B_state_dict": self.netD_B.state_dict(),
            "optimizer_G_state_dict": self.optimizer.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
        })
        return chkpt_data

    def load_checkpoint_data(self, chkpt_data: Dict, phase: str):
        """Load checkpoint data for all networks"""
        super().load_checkpoint_data(chkpt_data, phase)
        
        self.netG_A.load_state_dict(chkpt_data["netG_A_state_dict"])
        self.netG_B.load_state_dict(chkpt_data["netG_B_state_dict"])
        self.netD_A.load_state_dict(chkpt_data["netD_A_state_dict"])
        self.netD_B.load_state_dict(chkpt_data["netD_B_state_dict"])
        
        if phase.lower() == "train" and self.optimizer is not None:
            self.optimizer.load_state_dict(chkpt_data["optimizer_G_state_dict"])
            self.optimizer_D.load_state_dict(chkpt_data["optimizer_D_state_dict"])
