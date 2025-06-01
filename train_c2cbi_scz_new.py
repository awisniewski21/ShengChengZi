#!/usr/bin/env python3
"""
Character-to-Character (Bidirectional New) model training runner using the new base model architecture.
"""

from pathlib import Path

import torch
from diffusers import DDPMScheduler, UNet2DModel

from configs.c2cbi_scz import TrainConfig_C2CBi_SCZ
from core.dataset.datasets import get_dataloaders
from core.models import TrainModel_C2C_SCZ
from core.utils.train_utils import create_optimizer_and_scheduler
from core.utils.repo_utils import get_repo_dir


def main():
    """Main training function for ShengChengZi C2CBi model (new version)."""
    
    # Configuration with enhanced parameters for the new version
    ROOT_IMAGE_DIR = get_repo_dir() / Path("data/datasets/paired_64x64")  # Higher resolution
    METADATA_PATH = ROOT_IMAGE_DIR / "metadata.jsonl"

    cfg = TrainConfig_C2CBi_SCZ(
        image_size=64,  # Higher resolution
        train_batch_size=16,  # Smaller batch size for higher resolution
        eval_batch_size=8,
        num_epochs=200,  # More epochs for better convergence
        learning_rate=2e-5,  # Lower learning rate for stability
        save_image_epochs=5,
        save_model_epochs=20,
    )

    print(f"Starting ShengChengZi C2CBi training (new version) with config:")
    print(f"  Image size: {cfg.image_size}")
    print(f"  Batch size: {cfg.train_batch_size}")
    print(f"  Epochs: {cfg.num_epochs}")
    print(f"  Learning rate: {cfg.learning_rate}")

    # Data loaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        cfg, 
        root_image_dir=ROOT_IMAGE_DIR, 
        metadata_path=METADATA_PATH
    )

    print(f"Data loaders created:")
    print(f"  Train batches: {len(train_dataloader)}")
    print(f"  Val batches: {len(val_dataloader) if val_dataloader else 0}")
    print(f"  Test batches: {len(test_dataloader) if test_dataloader else 0}")

    # Enhanced model for higher resolution with more capacity
    model = UNet2DModel(
        sample_size=cfg.image_size,
        in_channels=2,  # Source image + noisy target image
        out_channels=1,  # Predicted noise for target
        layers_per_block=3,  # More layers per block for better features
        block_out_channels=(64, 128, 256, 512),  # More channels for higher capacity
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        attention_head_dim=8,  # Add attention for better feature interaction
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Enhanced model created with {total_params:,} trainable parameters")

    # Optimizer and schedulers
    optimizer, lr_scheduler = create_optimizer_and_scheduler(model, cfg, train_dataloader)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Create training model
    training_model = TrainModel_C2C_SCZ(
        config=cfg,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        noise_scheduler=noise_scheduler,
    )

    # Start training
    print("Starting enhanced training...")
    training_model.train()
    print("Training completed!")


if __name__ == "__main__":
    main()
