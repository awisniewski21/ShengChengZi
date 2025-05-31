from dataclasses import dataclass

from core.config.base_config import TrainingConfigBase


@dataclass
class TrainingConfigRand2Char(TrainingConfigBase):
    """ Model Params """
    task_name: str = "rand2char"                # Task name
    encoder_dim: int = 1024                     # Encoder dimensions
