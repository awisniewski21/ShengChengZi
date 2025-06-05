from dataclasses import dataclass

from configs import TrainConfigBase


@dataclass
class TrainConfig_R2C_Glyff(TrainConfigBase):
    """ Model Params """
    task_name: str = "rand2char"                # Task name
    
    """ Training Params """
    run_name_prefix: str = "train_r2c_glyff"    # Prefix for run names
    lr_warmup_steps: int = 500                  # Gradually increase lr to full value over first N steps
