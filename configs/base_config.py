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
    num_epochs: int = 100                       # Training epochs
    train_batch_size: int = 16                  # Training batch size (number of images)
    eval_batch_size: int = 16                   # Evaluation batch size (number of images)
    learning_rate: float = 1e-4                 # Model learning rate
    seed: int = 0                               # Seed for random number generators

    """ Environment """
    use_colab: bool = False                     # Whether running in Google Colab environment

    """ Logging """
    log_step_interval: int = 1                  # Log metrics every N steps
    eval_epoch_interval: int = 1                # Run validation every N epochs
    checkpoint_epoch_interval: int = 5          # Save model checkpoints every N epochs

    def __post_init__(self):
        self.repo_dir = Path(get_repo_dir())

    @property
    def output_dir(self) -> Path:
        """Get the full output directory path."""
        return self.repo_dir / "out"

    @property
    def root_image_dir(self) -> Path:
        """Get the root image directory path."""
        base_dir = self.repo_dir / "data" / "datasets" if not self.use_colab else Path("/content/datasets")
        dataset_type = "paired" if self.task_name in ["char2char", "char2char_bi"] else "unpaired"
        return base_dir / f"{dataset_type}_{self.image_size}x{self.image_size}"

    @property
    def image_metadata_path(self) -> Path:
        """Get the image metadata file path."""
        return self.root_image_dir / "metadata.jsonl"
