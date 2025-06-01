from dataclasses import dataclass

from configs.base_config import TrainingConfigBase


@dataclass
class TrainingConfigChar2CharBi(TrainingConfigBase):
    """ Model Params """
    task_name: str = "char2char_bi"             # Task name
    encoder_dim: int = 1024                     # Encoder dimensions
    
    """ Training Params (moved from base) """
    lr_warmup_steps: int = 500                  # Gradually increase lr to full value over first N steps
