from dataclasses import dataclass

from configs import TrainConfigBase


@dataclass
class TrainConfig_C2CBi_SCZ(TrainConfigBase):
    """ Model Params """
    task_name: str = "char2char_bi"             # Task name
    
    """ Training Params """
    lr_warmup_steps: int = 500                  # Gradually increase lr to full value over first N steps
