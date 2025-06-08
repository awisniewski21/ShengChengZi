import torch
import torch.nn as nn

from core.configs import TrainConfig_C2C_Pix2Pix
from cyclegan_and_pix2pix.networks import define_D, define_G


class Pix2PixNetwork(nn.Module):
    """
    Pix2Pix network wrapper that contains all sub-networks
    """
    def __init__(self, config: TrainConfig_C2C_Pix2Pix):
        super().__init__()
        self.config = config

        common_kwargs = dict(
            norm=config.norm,
            init_type=config.init_type,
            init_gain=config.init_gain,
            gpu_ids=[],
        )

        # Define generator
        self.netG = define_G(
            input_nc=config.input_nc,
            output_nc=config.output_nc,
            ngf=config.ngf,
            netG=config.netG,
            use_dropout=not config.no_dropout,
            **common_kwargs,
        )

        # Define discriminator - takes both input and output images
        self.netD = define_D(
            input_nc=config.input_nc + config.output_nc,
            ndf=config.ndf,
            netD=config.netD,
            n_layers_D=config.n_layers_D,
            **common_kwargs,
        )

    def forward(self, real_A: torch.Tensor) -> torch.Tensor:
        """ Run forward pass for Pix2Pix """
        return self.netG(real_A) # G(A)

    def set_G_requires_grad(self, requires_grad: bool = False):
        """ Set requires_grad for all generators """
        for param in self.netG.parameters():
            param.requires_grad = requires_grad

    def set_D_requires_grad(self, requires_grad: bool = False):
        """ Set requires_grad for all discriminators """
        for param in self.netD.parameters():
            param.requires_grad = requires_grad