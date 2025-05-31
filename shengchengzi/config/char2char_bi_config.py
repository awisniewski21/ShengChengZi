from dataclasses import dataclass

from core.config.base_config import TrainingConfigBase


@dataclass
class TrainingConfigChar2CharBi(TrainingConfigBase):
    """ Model Params """
    task_name: str = "char2char_bi"             # Task name
    encoder_dim: int = 1024                     # Encoder dimensions
