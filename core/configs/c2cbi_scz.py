from dataclasses import dataclass

from core.configs import TrainConfigBase


@dataclass
class TrainConfig_C2CBi_SCZ(TrainConfigBase):
    """ Model Params """
    dataset_task: str = "char2char_bi"          # Dataset task name
    
    """ Training Params """
    run_name_prefix: str = "train_c2cbi_scz"    # Prefix for run names
    lr_warmup_steps: int = 500                  # Gradually increase lr to full value over first N steps
