import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

from configs.c2c_palette import TrainConfig_C2C_Palette


def parse_dataclass_args(root_image_dir: str = None, phase: str = "train", batch: int = None, gpu_ids: str = None, debug: bool = False, **kwargs) -> TrainConfig_C2C_Palette:
    """
    Create palette configuration from dataclass instead of JSON.
    """
    # Create config with custom data root if provided
    cfg_kwargs = {}
    if root_image_dir:
        cfg_kwargs["root_image_dir"] = root_image_dir
    
    # Apply any additional overrides
    cfg_kwargs.update(kwargs)
    
    # Create config instance
    cfg = TrainConfig_C2C_Palette(**cfg_kwargs)
    
    # Apply command line overrides
    cfg.phase = phase
    if gpu_ids is not None:
        cfg.gpu_ids = [int(id) for id in gpu_ids.split(",")]
    if batch is not None:
        if phase == "train":
            cfg.train_batch_size = batch
        else:
            cfg.eval_batch_size = batch

    # Set CUDA environment
    cfg.distributed = cfg.gpu_ids is not None and len(cfg.gpu_ids) > 1

    # Update experiment name
    prefix = "debug" if debug else cfg.phase
    cfg.name = f"{prefix}_{cfg.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Update training options if debugging
    if "debug" in cfg.name:
        # Apply debug settings
        cfg.save_model_epochs = 1
        cfg.log_iter = 1

    return cfg