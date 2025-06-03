import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

from configs import TrainConfigBase


def setup_train(cfg: TrainConfigBase, task_prefix: str) -> Tuple[torch.device, str, str, SummaryWriter]:
    """
    Handles necessary setup for training
    Sets the device, creates output directories, and initializes TensorBoard writer
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    else:
        device = torch.device("cpu")
        warnings.warn("CUDA and MPS are not available - defaulting to CPU")

    if cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)

    run_name = f"{task_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = str(cfg.output_dir / "logs" / run_name)
    writer = SummaryWriter(log_dir=log_dir)

    return device, run_name, log_dir, writer
