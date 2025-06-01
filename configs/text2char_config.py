from dataclasses import dataclass

from configs.base_config import TrainingConfigBase


@dataclass
class TrainingConfigText2Char(TrainingConfigBase):
    """ Model Params """
    task_name: str = "text2char"                # Task name
    text_encoder: str = "google-t5/t5-small"    # Text encoder model name
    encoder_dim: int = 1024                     # Encoder dimensions
    
    """ Training Params (moved from base) """
    lr_warmup_steps: int = 500                  # Gradually increase lr to full value over first N steps
