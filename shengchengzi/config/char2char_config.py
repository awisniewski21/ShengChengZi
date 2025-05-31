from dataclasses import dataclass

from core.config.base_config import TrainingConfigBase


@dataclass
class TrainingConfigChar2Char(TrainingConfigBase):
    """ Model Params """
    encoder_dim: int = 1024                     # Encoder dimensions
