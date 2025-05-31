#!/usr/bin/env python3
"""
Random-to-Character model training runner.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from core.utils.repo_utils import get_repo_dir
from core.utils.train_utils import get_dataloader
from glyffuser.config.rand2char_config import TrainingConfigRand2Char
from glyffuser.utils.eval_utils import DiffusionPipelineRand2Char


def train_loop(
    cfg: TrainingConfigRand2Char,
    train_dataloader: DataLoader,
    model: UNet2DModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    noise_scheduler: DDPMScheduler,
    inference_scheduler: DPMSolverMultistepScheduler,
):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = model.to(device)

    # Tensorboard logging
    if cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)
    run_name = f"train_glyffuser_rand2char_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = str(Path(cfg.output_dir) / "logs" / run_name)
    writer = SummaryWriter(log_dir=log_dir)

    # Train loop
    global_step = 0
    for epoch in range(cfg.num_epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")

        model.train()
        for b_imgs in pbar:
            b_imgs = b_imgs.to(device)

            noise = torch.randn_like(b_imgs)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b_imgs.shape[0],), device=device).long()
            b_imgs_noisy = noise_scheduler.add_noise(b_imgs, noise, timesteps)

            optimizer.zero_grad()
            noise_pred = model(b_imgs_noisy, timesteps).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            pbar.set_postfix(**logs)
            writer.add_scalar("Loss/train", logs["loss"], global_step)
            writer.add_scalar("LR", logs["lr"], global_step)
            global_step += 1

        model.eval()
        pipeline = DiffusionPipelineRand2Char(unet=model, scheduler=noise_scheduler)
        pipeline.set_progress_bar_config(desc="Generating evaluation image grid...")

        # Save model checkpoint
        if epoch % cfg.save_model_epochs == 0 or epoch == cfg.num_epochs - 1:
            if epoch > 0:
                pipeline.save_pretrained(str(Path(cfg.output_dir) / "models" / run_name / f"epoch_{epoch}"))
                pipeline.save_pretrained(str(Path(cfg.output_dir) / "models" / run_name / f"latest"))

        # Evaluate and log images
        if epoch % cfg.save_image_epochs == 0 or epoch == cfg.num_epochs - 1:
            img_grid = pipeline.evaluate_texts_to_image_grid(batch_size=cfg.eval_batch_size, output_type="numpy", seed=cfg.seed)
            writer.add_images("eval_imgs", img_grid, global_step, dataformats="NHWC")
        writer.flush()
    writer.close()


def main():
    # Try to mount Google Drive (for Colab compatibility)
    try:
        from google.colab import drive
        drive.mount("/content/gdrive/")
    except:
        pass

    # Configuration
    ROOT_IMAGE_DIR = get_repo_dir() / Path("data/data")
    
    cfg = TrainingConfigRand2Char(
        image_size=32,
        train_batch_size=32,
        eval_batch_size=16,
        encoder_dim=512,
        save_image_epochs=1,
        save_model_epochs=5,
    )

    # Data loader
    train_dataloader = get_dataloader(cfg, ROOT_IMAGE_DIR)

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
    print(f"Total Model Parameters: {total_params:,}")

    # Optimizer and schedulers
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=len(train_dataloader) * cfg.num_epochs,
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    inference_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000)

    # Train the model
    print("Starting training...")
    train_loop(cfg, train_dataloader, model, optimizer, lr_scheduler, noise_scheduler, inference_scheduler)
    print("Training completed!")


if __name__ == "__main__":
    main()
