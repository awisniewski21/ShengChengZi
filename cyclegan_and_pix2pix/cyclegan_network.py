from typing import NamedTuple

import torch
import torch.nn as nn

from core.configs import TrainConfig_C2CBi_CycleGAN
from cyclegan_and_pix2pix.networks import define_D, define_G


CycleGANOutput = NamedTuple("CycleGANOutput", [
    ("fake_B", torch.Tensor), # G_A(A)
    ("rec_A", torch.Tensor),  # G_B(G_A(A))
    ("fake_A", torch.Tensor), # G_B(B)
    ("rec_B", torch.Tensor),  # G_A(G_B(B))
])

class CycleGANNetwork(nn.Module):
    """
    CycleGAN network wrapper that contains all sub-networks
    """
    def __init__(self, config: TrainConfig_C2CBi_CycleGAN):
        super().__init__()
        self.config = config

        common_kwargs = dict(
            norm=config.norm,
            init_type=config.init_type,
            init_gain=config.init_gain,
            gpu_ids=[],
        )

        # Define generators
        common_G_kwargs = dict(
            ngf=config.ngf,
            netG=config.netG,
            use_dropout=not config.no_dropout,
            **common_kwargs,
        )
        self.netG_A = define_G(input_nc=config.input_nc, output_nc=config.output_nc, **common_G_kwargs)
        self.netG_B = define_G(input_nc=config.output_nc, output_nc=config.input_nc, **common_G_kwargs)

        # Define discriminators
        common_D_kwargs = dict(
            ndf=config.ndf,
            netD=config.netD,
            n_layers_D=config.n_layers_D,
            **common_kwargs,
        )
        self.netD_A = define_D(input_nc=config.output_nc, **common_D_kwargs)
        self.netD_B = define_D(input_nc=config.input_nc, **common_D_kwargs)

    def forward(self, real_A: torch.Tensor, real_B: torch.Tensor) -> CycleGANOutput:
        """ Run forward pass for CycleGAN """
        fake_B = self.netG_A(real_A) # G_A(A)
        rec_A = self.netG_B(fake_B)  # G_B(G_A(A))
        fake_A = self.netG_B(real_B) # G_B(B)
        rec_B = self.netG_A(fake_A)  # G_A(G_B(B))

        return CycleGANOutput(fake_B, rec_A, fake_A, rec_B)

    def set_G_requires_grad(self, requires_grad: bool = False):
        """ Set requires_grad for all generators """
        for param in self.netG_A.parameters():
            param.requires_grad = requires_grad
        for param in self.netG_B.parameters():
            param.requires_grad = requires_grad

    def set_D_requires_grad(self, requires_grad: bool = False):
        """ Set requires_grad for all discriminators """
        for param in self.netD_A.parameters():
            param.requires_grad = requires_grad
        for param in self.netD_B.parameters():
            param.requires_grad = requires_grad
