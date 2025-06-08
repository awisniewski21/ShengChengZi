from dataclasses import dataclass

from configs import TrainConfigBase


@dataclass
class TrainConfig_C2C_CycleGAN(TrainConfigBase):
    """ Dataset """
    task_name: str = "char2char"                # Task name

    """ Training Params """
    run_name_prefix: str = "train_c2c_cyclegan" # Prefix for run names

    """ Model Params """
    # Network Architecture
    netG: str = "resnet_9blocks"                # Generator architecture
    netD: str = "basic"                         # Discriminator architecture
    ngf: int = 64                               # Generator filters in last conv layer
    ndf: int = 64                               # Discriminator filters in first conv layer
    n_layers_D: int = 3                         # Number of discriminator layers
    norm: str = "instance"                      # Normalization type
    init_type: str = "normal"                   # Network initialization
    init_gain: float = 0.02                     # Initialization scaling factor
    no_dropout: bool = True                     # No dropout for generator

    # Loss Weights
    lambda_A: float = 10.0                      # Weight for cycle loss (A -> B -> A)
    lambda_B: float = 10.0                      # Weight for cycle loss (B -> A -> B)
    lambda_identity: float = 0.5                # Weight for identity mapping loss

    # GAN Loss
    gan_mode: str = "lsgan"                     # GAN loss type
    pool_size: int = 50                         # Image buffer size

    # Learning Rate
    lr_policy: str = "linear"                   # Learning rate decay policy
    lr_decay_iters: int = 50                    # Linear decay iterations

    # Input/Output Channels
    input_nc: int = 1                           # Input image channels (grayscale)
    output_nc: int = 1                          # Output image channels (grayscale)
