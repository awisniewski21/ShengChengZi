from dataclasses import dataclass
from pathlib import Path

from utils.utils import get_repo_dir


@dataclass
class TrainingConfigBase:
    """ Dataset """
    image_size: int = 128                       # Image resolution

    """ Training Params """
    num_epochs: int = 100                       # Training epochs
    train_batch_size: int = 16                  # Training batch size (number of images)
    eval_batch_size: int = 16                   # Evaluation batch size (number of images)
    learning_rate: float = 1e-4                 # Model learning rate
    lr_warmup_steps: int = 500                  # Gradually increase lr to full value over first N steps
    seed: int = 0                               # Seed for random number generators

    """ Logging """
    output_dir: str = "out"                     # Output directory
    overwrite_output_dir: bool = True           # Overwrite old models with the same name
    save_image_epochs: int = 10                 # Save training images every N epochs
    save_model_epochs: int = 30                 # Save model checkpoints every N epochs


    def __post_init__(self):
        self.output_dir = str(Path(get_repo_dir()) / self.output_dir)