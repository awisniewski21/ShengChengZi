from dataclasses import dataclass

from core.configs import TrainConfigBase


@dataclass
class TrainConfig_T2C_Glyff(TrainConfigBase):
    """ Model Params """
    dataset_task: str = "text2char"             # Dataset task name
    text_encoder: str = "google-t5/t5-small"    # Text encoder model name
    encoder_dim: int = 1024                     # Encoder dimensions

    """ Training Params """
    run_name_prefix: str = "train_t2c_glyff"    # Prefix for run names
    lr_warmup_steps: int = 500                  # Gradually increase lr to full value over first N steps
