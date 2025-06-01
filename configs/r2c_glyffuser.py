from dataclasses import dataclass

from configs import TrainConfigBase


@dataclass
class TrainConfig_R2C_Glyff(TrainConfigBase):
    """ Model Params """
    task_name: str = "rand2char"                # Task name
    
    """ Training Params """
    lr_warmup_steps: int = 500                  # Gradually increase lr to full value over first N steps
