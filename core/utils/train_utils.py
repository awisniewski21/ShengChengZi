#!/usr/bin/env python3
"""
Common training utilities to reduce code duplication across training scripts.
Only includes larger utility functions that provide meaningful code reduction.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs.base_config import TrainConfigBase


def setup_training_environment(cfg: TrainConfigBase, task_prefix: str) -> Tuple[torch.device, str, str, SummaryWriter]:
    """
    Set up the training environment with device, directories, and tensorboard writer.
    
    Args:
        cfg: Training configuration
        task_prefix: Prefix for the run name (e.g., "train_glyffuser_rand2char")
    
    Returns:
        Tuple of (device, run_name, log_dir, writer)
    """
    # Get best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Create output directory
    if cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Generate unique run name with timestamp
    run_name = f"{task_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = str(Path(cfg.output_dir) / "logs" / run_name)
    
    # Set up tensorboard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    return device, run_name, log_dir, writer


def create_optimizer_and_scheduler(model: torch.nn.Module, cfg: TrainConfigBase, train_dataloader: DataLoader):
    """
    Create optimizer and learning rate scheduler for training.
    
    Args:
        model: The model to optimize
        cfg: Training configuration
        train_dataloader: Training dataloader for calculating total training steps
    
    Returns:
        Tuple of (optimizer, lr_scheduler)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=len(train_dataloader) * cfg.num_epochs,
    )
    
    return optimizer, lr_scheduler





