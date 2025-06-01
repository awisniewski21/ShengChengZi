#!/usr/bin/env python3
"""
Random-to-Character model training runner using the new base model architecture.
"""

from pathlib import Path

import torch
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, UNet2DModel

from configs.r2c_glyffuser import TrainConfig_R2C_Glyff
from core.dataset.datasets import get_dataloaders
from core.models import TrainModel_R2C_Glyffuser
from core.utils.train_utils import create_optimizer_and_scheduler
from core.utils.repo_utils import get_repo_dir


def main():
    """Main training function for Glyffuser R2C model."""
    
    # Configuration
    ROOT_IMAGE_DIR = get_repo_dir() / Path("data/datasets/unpaired_32x32")
    METADATA_PATH = ROOT_IMAGE_DIR / "metadata.jsonl"

    cfg = TrainConfig_R2C_Glyff(
        image_size=32,
        train_batch_size=32,
        eval_batch_size=16,
        save_image_epochs=1,
        save_model_epochs=5,
    )

    print(f"Starting Glyffuser R2C training with config:")
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

    # Model
    model = UNet2DModel(
        sample_size=cfg.image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64, 128, 128),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} trainable parameters")

    # Optimizer and schedulers
    optimizer, lr_scheduler = create_optimizer_and_scheduler(model, cfg, train_dataloader)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    inference_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000)

    # Create training model
    training_model = TrainModel_R2C_Glyffuser(
        config=cfg,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        noise_scheduler=noise_scheduler,
        inference_scheduler=inference_scheduler,
    )

    # Start training
    print("Starting training...")
    training_model.train()
    print("Training completed!")


if __name__ == "__main__":
    main()
