from dataclasses import dataclass

from core.config.base_config import TrainingConfigBase


@dataclass
class TrainingConfigText2Char(TrainingConfigBase):
    """ Model Params """
    text_encoder: str = "google-t5/t5-small"    # Text encoder model name
    encoder_dim: int = 1024                     # Encoder dimensions
