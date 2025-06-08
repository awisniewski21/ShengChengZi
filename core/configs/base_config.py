from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

from core.utils.repo_utils import get_repo_dir


@dataclass
class TrainConfigBase:
    """ Dataset """
    task_name: str                              # Task name ("rand2char", "text2char", "char2char", or "char2char_bi")
    image_size: int = 64                        # Image resolution

    """ Dataset Split """
    validation_split: float | int = 16          # Validation split (0.0-1.0 or int for absolute count)
    test_split: float | int = 0.05              # Test split (0.0-1.0 or int for absolute count)

    """ Training Params """
    run_name_prefix: str = "train"              # Prefix for run names
    num_epochs: int = 100                       # Training epochs
    train_batch_size: int = 16                  # Training batch size (number of images)
    eval_batch_size: int = 16                   # Evaluation batch size (number of images)
    learning_rate: float = 1e-4                 # Model learning rate
    load_checkpoint_path: str | None = None     # Optional path to load model checkpoint from
    seed: int = 0                               # Seed for random number generators

    """ Environment """
    use_colab: bool = False                     # Whether running in Google Colab environment

    """ Logging """
    log_step_interval: int = 1                  # Log step metrics every N steps
    eval_epoch_interval: int = 1                # Run validation every N epochs
    checkpoint_epoch_interval: int = 50         # Save model checkpoints every N epochs

    @classmethod
    def from_dict(cls, config_dict: Dict) -> TrainConfigBase:
        """ Create a configuration instance from a dictionary """
        return cls(**config_dict)

    @property
    def output_dir(self) -> Path:
        """ Get the full output directory path """
        return get_repo_dir() / "out"

    @property
    def root_image_dir(self) -> Path:
        """ Get the root image directory path """
        base_data_dir = get_repo_dir() / "data" if not self.use_colab else Path("/content")
        dataset_type = "paired" if self.task_name in ["char2char", "char2char_bi"] else "unpaired"
        return base_data_dir / "datasets" / f"{dataset_type}_{self.image_size}x{self.image_size}"

    @property
    def image_metadata_path(self) -> Path:
        """ Get the image metadata file path """
        return self.root_image_dir / "metadata.jsonl"

    def to_dict(self) -> Dict:
        """ Convert the configuration to a dictionary """
        return asdict(self)