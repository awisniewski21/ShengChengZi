from dataclasses import dataclass

from configs.base_config import TrainConfigBase


@dataclass
class TrainConfig_R2C_Glyff(TrainConfigBase):
    """ Model Params """
    task_name: str = "rand2char"                # Task name
    encoder_dim: int = 1024                     # Encoder dimensions
    
    """ Training Params (moved from base) """
    lr_warmup_steps: int = 500                  # Gradually increase lr to full value over first N steps
