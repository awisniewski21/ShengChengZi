#!/usr/bin/env python3
"""
Palette model training runner using the merged Palette model.
"""

import os
import warnings
from pathlib import Path

import click
import torch
import torch.multiprocessing as mp

from palette.utils.device_utils import set_seed
from palette.utils.parser import parse_dataclass_args
from configs.c2c_palette import TrainConfig_C2C_Palette
from core.models import TrainModel_C2C_Palette
from core.dataset.datasets import get_dataloaders


def main_worker(gpu: int, ngpus_per_node: int, config: TrainConfig_C2C_Palette):
    """
    Main function to run on each thread / GPU using the merged Palette model.
    """
    if config.local_rank is None:
        config.local_rank = config.global_rank = gpu

    print(f"Starting Palette C2C training on GPU {gpu}")
    
    # Create data loaders using get_dataloaders directly
    phase_loader, val_loader, test_loader = get_dataloaders(
        config, 
        config.root_image_dir,
        metadata_path=str(Path(config.root_image_dir) / "metadata.jsonl")
    )
    
    # Create Palette network
    from palette.models.palette_network import PaletteNetwork
    network = PaletteNetwork(config=config)
    
    # Initialize network weights
    if hasattr(network, 'init_weights'):
        network.init_weights()
    
    # Create loss function
    from palette.models.loss import mse_loss
    loss_fn = mse_loss
    
    # Create optimizer (will be properly configured in the model)
    optimizer_config = {"lr": config.learning_rate}
    
    # Create our integrated Palette model
    training_model = TrainModel_C2C_Palette(
        config=config,
        train_dataloader=phase_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        networks=[network],
        losses=[loss_fn],
        sample_num=getattr(config, 'sample_num', 8),
        optimizers=[optimizer_config],
        metrics=None,
        ema_scheduler=None,
    )
    
    # Start training
    if config.phase == "train":
        training_model.train()
    else:
        training_model.test()


@click.command()
@click.option("--resume_state", "-r", default=None, help="Resume training state path")
@click.option("--local_rank", "-l", default=None, type=int, help="Local rank for distributed training")
@click.option("--num_gpus", "-n", default=1, type=int, help="Number of GPUs to use")
def main(resume_state, local_rank, num_gpus):
    """
    Main entry point for Palette training with the new base model architecture.
    Uses dataclass-based configuration instead of JSON config files.
    """
    print("Starting Palette training with dataclass configuration")
    
    # Parse configuration from command line arguments and dataclass defaults
    config_obj = parse_dataclass_args()  # Returns TrainConfig_C2C_Palette
    if resume_state is not None:
        config_obj.resume_state = resume_state
    if local_rank is not None:
        config_obj.local_rank = local_rank
    
    print(f"Configuration loaded:")
    print(f"  Phase: {config_obj.phase}")
    print(f"  Name: {config_obj.name}")
    print(f"  Distributed: {config_obj.distributed}")
    print(f"  Number of GPUs: {num_gpus}")

    # Set multiprocessing method
    if hasattr(mp, "_supports_context") and mp._supports_context:
        mp.set_start_method("spawn", force=True)

    if config_obj.distributed and num_gpus > 1:
        print(f"Launching distributed training on {num_gpus} GPUs")
        config_obj.world_size = num_gpus
        config_obj.init_method = f"tcp://127.0.0.1:23456"  # Use default port
        mp.spawn(main_worker, nprocs=num_gpus, args=(num_gpus, config_obj))
    else:
        print("Running single GPU training")
        config_obj.world_size = 1
        config_obj.distributed = False
        main_worker(0, 1, config_obj)


if __name__ == "__main__":
    main()
