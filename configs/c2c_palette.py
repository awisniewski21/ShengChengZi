from dataclasses import dataclass, field
from typing import List

from configs import TrainConfigBase


@dataclass
class TrainConfig_C2C_Palette(TrainConfigBase):
    """ Dataset """
    task_name: str = "char2char"                # Task name

    """ Training Params """
    run_name_prefix: str = "train_c2c_palette"  # Prefix for run names

    """ Model Params """
    # UNet
    module_name: str = "guided_diffusion"       # UNet module type
    init_type: str = "kaiming"                  # Weight initialization
    in_channel: int = 2                         # Input channels
    out_channel: int = 1                        # Output channels  
    inner_channel: int = 64                     # Base channel count
    channel_mults: List[int] = field(           # Channel multipliers
        default_factory=lambda: [1, 2, 4, 8]
    )
    attn_res: List[int] = field(                # Attention resolutions
        default_factory=lambda: [16]
    )
    num_head_channels: int = 32                 # Attention head channels
    res_blocks: int = 2                         # Residual blocks per level
    dropout: float = 0.2                        # Dropout rate

    # Diffusion Schedule  
    schedule: str = "linear"                    # Beta schedule type
    n_timestep_train: int = 1000                # Training timesteps
    n_timestep_test: int = 500                  # Testing timesteps
    linear_start_train: float = 1e-6            # Training schedule start
    linear_end_train: float = 0.01              # Training schedule end
    linear_start_test: float = 1e-4             # Testing schedule start
    linear_end_test: float = 0.09               # Testing schedule end

    # EMA
    ema_enabled: bool = True                    # Enable EMA
    ema_start: int = 1                          # EMA start iteration
    ema_iter: int = 1                           # EMA update frequency
    ema_decay: float = 0.9999                   # EMA decay rate

    """ Logging """
    sample_num: int = 8                         # Number of intermediate images to output from denoising process
