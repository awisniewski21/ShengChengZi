from dataclasses import dataclass

from core.configs import TrainConfigBase


@dataclass
class TrainConfig_C2C_Pix2Pix(TrainConfigBase):
    """ Dataset """
    task_name: str = "char2char"                # Task name

    """ Training Params """
    run_name_prefix: str = "train_c2c_pix2pix"  # Prefix for run names

    """ Model Params """
    # Network Architecture
    netG: str = "resnet_6blocks"                # Generator architecture (better for 64x64 than unet_256)
    netD: str = "basic"                         # Discriminator architecture
    ngf: int = 64                               # Generator filters in last conv layer
    ndf: int = 64                               # Discriminator filters in first conv layer
    n_layers_D: int = 3                         # Number of discriminator layers
    norm: str = "batch"                         # Normalization type
    init_type: str = "normal"                   # Network initialization
    init_gain: float = 0.02                     # Initialization scaling factor
    no_dropout: bool = False                    # Use dropout for generator

    # Loss Weights
    lambda_L1: float = 100.0                    # Weight for L1 loss

    # GAN Loss
    gan_mode: str = "vanilla"                   # GAN loss type
    pool_size: int = 0                          # No image buffer for pix2pix

    # Learning Rate
    lr_policy: str = "linear"                   # Learning rate decay policy
    lr_decay_iters: int = 50                    # Linear decay iterations

    # Input/Output Channels
    input_nc: int = 1                           # Input image channels (grayscale)
    output_nc: int = 1                          # Output image channels (grayscale)
