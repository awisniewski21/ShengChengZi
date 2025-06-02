from dataclasses import dataclass
from pathlib import Path

from core.utils.repo_utils import get_repo_dir


@dataclass
class TrainConfigBase:
    """ Dataset """
    task_name: str                              # Task name ("rand2char", "text2char", "char2char", or "char2char_bi")
    image_size: int = 32                        # Image resolution

    """ Dataset Split """
    validation_split: float | int = 16          # Validation split (0.0-1.0 or int for absolute count)
    test_split: float | int = 16                # Test split (0.0-1.0 or int for absolute count)

    """ Training Params """
    num_epochs: int = 20                        # Training epochs
    train_batch_size: int = 16                  # Training batch size (number of images)
    eval_batch_size: int = 16                   # Evaluation batch size (number of images)
    learning_rate: float = 1e-4                 # Model learning rate
    seed: int = 0                               # Seed for random number generators

    """ Logging """
    output_dir: str = "out"                     # Output directory
    log_step_interval: int = 1                  # Log metrics every N steps
    eval_epoch_interval: int = 1                # Run validation every N epochs
    checkpoint_epoch_interval: int = 5          # Save model checkpoints every N epochs

    def __post_init__(self):
        self.output_dir = str(Path(get_repo_dir()) / self.output_dir)
