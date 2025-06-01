from pathlib import Path

import torch
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup

from configs import TrainConfig_R2C_Glyff
from core.dataset.datasets import get_dataloaders
from core.models import TrainModel_R2C_Glyffuser
from core.utils.repo_utils import get_repo_dir


def main():
    ROOT_IMAGE_DIR = get_repo_dir() / Path("data/datasets/unpaired_32x32")
    METADATA_PATH = ROOT_IMAGE_DIR / "metadata.jsonl"

    # Config
    cfg = TrainConfig_R2C_Glyff(
        image_size=32,
        train_batch_size=32,
        eval_batch_size=16,
        eval_epoch_interval=1,
        checkpoint_epoch_interval=5,
    )

    print(f"Starting Rand2Char training with Glyffuser model:")
    print(f"    Image size: {cfg.image_size}")
    print(f"    Batch size: {cfg.train_batch_size}")
    print(f"    Epochs: {cfg.num_epochs}")
    print(f"    Learning rate: {cfg.learning_rate}")

    # Dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        cfg, 
        root_image_dir=ROOT_IMAGE_DIR, 
        metadata_path=METADATA_PATH
    )

    # Network
    net = UNet2DModel(
        sample_size=cfg.image_size,
        in_channels=1,
        out_channels=1,
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels=(32, 64, 128, 128),
        layers_per_block=2,
    )

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable network parameters: {total_params:,}")

    # Optimizer and schedulers
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=len(train_dataloader) * cfg.num_epochs,
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    inference_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000)

    # Model
    training_model = TrainModel_R2C_Glyffuser(
        config=cfg,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        net=net,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        noise_scheduler=noise_scheduler,
        inference_scheduler=inference_scheduler,
    )

    # Training loop
    training_model.train()


if __name__ == "__main__":
    main()
