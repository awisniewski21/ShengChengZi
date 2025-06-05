from dataclasses import dataclass

from configs import TrainConfigBase


@dataclass
class TrainConfig_C2CBi_SCZ(TrainConfigBase):
    """ Model Params """
    task_name: str = "char2char_bi"             # Task name
    
    """ Training Params """
    run_name_prefix: str = "train_c2cbi_scz"    # Prefix for run names
    lr_warmup_steps: int = 500                  # Gradually increase lr to full value over first N steps
