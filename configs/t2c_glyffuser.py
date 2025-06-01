from dataclasses import dataclass

from configs import TrainConfigBase


@dataclass
class TrainConfig_T2C_Glyff(TrainConfigBase):
    """ Model Params """
    task_name: str = "text2char"                # Task name
    text_encoder: str = "google-t5/t5-small"    # Text encoder model name
    encoder_dim: int = 1024                     # Encoder dimensions
    
    """ Training Params """
    lr_warmup_steps: int = 500                  # Gradually increase lr to full value over first N steps
