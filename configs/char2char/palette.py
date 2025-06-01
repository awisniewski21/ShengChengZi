from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from configs.base_config import TrainConfigBase


@dataclass
class TrainConfig_C2C_Palette(TrainConfigBase):
    """Palette model configuration extending base config"""

    """ Dataset """
    task_name: str = "char2char"                # Task name
    root_image_dir: str = "data/datasets/paired_64x64"
    validation_split: int = 2                   # Validation split size

    """ Training Params """
    name: str = "palette_char2char"             # Experiment name
    val_epoch: int = 1                          # Validation frequency (in epochs)

    loss_function: str = "mse_loss"             # Loss function
    metrics: List[str] = field(default_factory=lambda: ["mae"])  # Evaluation metrics

    """ Model Params """
    sample_num: int = 8                         # Number of sampling steps

    # UNet Architecture
    in_channel: int = 2                         # Input channels
    out_channel: int = 1                        # Output channels  
    inner_channel: int = 64                     # Base channel count
    channel_mults: List[int] = field(default_factory=lambda: [1, 2, 4, 8])  # Channel multipliers
    attn_res: List[int] = field(default_factory=lambda: [16])               # Attention resolutions
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

    # Network Configuration
    module_name: str = "guided_diffusion"       # UNet module type
    init_type: str = "kaiming"                  # Weight initialization

    weight_decay: float = 0.0                   # Weight decay

    # EMA Configuration
    ema_enabled: bool = True                    # Enable EMA
    ema_start: int = 1                          # EMA start iteration
    ema_iter: int = 1                           # EMA update frequency
    ema_decay: float = 0.9999                   # EMA decay rate

    """ Logging """
    tensorboard: bool = True                    # Enable tensorboard
    log_iter: int = 10                          # Logging frequency
    log_dir: str = "logs"                       # Log directory
    model_dir: str = "models"                   # Model directory  # REMOVABLE: Not used in current code
    resume_state: Optional[str] = None          # Resume from checkpoint

    """ System """
    phase: str = "train"                        # Training phase ("train" or "test")
    gpu_ids: List[int] = field(default_factory=list)  # GPU IDs to use
    distributed: bool = False                   # Distributed training
    local_rank: Optional[int] = None            # Local GPU rank
    global_rank: int = 0                        # Global rank
    world_size: int = 1                         # World size
    init_method: Optional[str] = None           # Distributed init method

    def __post_init__(self):
        super().__post_init__()

        self.root_image_dir = str(Path(self.output_dir) / self.root_image_dir)
        self.log_dir = str(Path(self.output_dir) / self.log_dir)
        self.model_dir = str(Path(self.output_dir) / self.model_dir)
