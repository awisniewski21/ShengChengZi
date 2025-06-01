#!/usr/bin/env python3
"""
Text-to-Character model training runner.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel  # NOQA
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from configs.t2c_glyffuser import TrainConfig_T2C_Glyff  # NOQA
from core.dataset.datasets import get_dataloaders
from core.utils.eval_utils import evaluate_test_set
from core.utils.repo_utils import get_repo_dir
from core.utils.train_utils import create_optimizer_and_scheduler, setup_training_environment  # NOQA
from core.utils.eval_utils import DiffusionPipeline_T2C_Glyff


def train_loop(
    cfg: TrainConfig_T2C_Glyff,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model: UNet2DConditionModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    noise_scheduler: DDPMScheduler,
    inference_scheduler: DPMSolverMultistepScheduler,
):
    # Setup training environment
    device, run_name, log_dir, writer = setup_training_environment(cfg, "train_glyffuser_text2char")
    
    model = model.to(device)

    # Train loop
    global_step = 0
    for epoch in range(cfg.num_epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")

        model.train()
        for b_imgs, b_texts_embed, b_masks in pbar:
            b_imgs = b_imgs.to(device)
            b_texts_embed = b_texts_embed.to(device)
            b_masks = b_masks.to(device)

            noise = torch.randn_like(b_imgs)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b_imgs.shape[0],), device=device).long()
            b_imgs_noisy = noise_scheduler.add_noise(b_imgs, noise, timesteps)

            optimizer.zero_grad()
            noise_pred = model(b_imgs_noisy, timesteps, encoder_hidden_states=b_texts_embed).sample
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
        pipeline = DiffusionPipeline_T2C_Glyff(unet=model, scheduler=noise_scheduler)
        pipeline.set_progress_bar_config(desc="Generating evaluation image grid...")

        # Validation evaluation
        if len(val_dataloader) > 0:
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_imgs, val_texts_embed, val_masks in val_dataloader:
                    val_imgs = val_imgs.to(device)
                    val_texts_embed = val_texts_embed.to(device)
                    val_masks = val_masks.to(device)
                    
                    noise = torch.randn_like(val_imgs)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (val_imgs.shape[0],), device=device).long()
                    val_imgs_noisy = noise_scheduler.add_noise(val_imgs, noise, timesteps)
                    noise_pred = model(val_imgs_noisy, timesteps, encoder_hidden_states=val_texts_embed).sample
                    val_loss += torch.nn.functional.mse_loss(noise_pred, noise).item()
                    val_steps += 1
            
            avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
            writer.add_scalar("Loss/validation", avg_val_loss, global_step)
            print(f"Epoch {epoch} - Validation Loss: {avg_val_loss:.4f}")

        # Save model checkpoint
        if epoch % cfg.save_model_epochs == 0 or epoch == cfg.num_epochs - 1:
            if epoch > 0:
                save_dir = Path(cfg.output_dir) / "models" / run_name
                save_dir.mkdir(parents=True, exist_ok=True)
                pipeline.save_pretrained(str(save_dir / f"epoch_{epoch}"))
                pipeline.save_pretrained(str(save_dir / "latest"))

        # Evaluate and log images
        if epoch % cfg.save_image_epochs == 0 or epoch == cfg.num_epochs - 1:
            texts = ["love", "hate", "hot", "cold", "one", "two", "up", "down", "day", "night", "top", "bottom", "summer", "winter", "white", "black"]
            img_grid = pipeline.evaluate_texts_to_image_grid(texts, batch_size=cfg.eval_batch_size, output_type="numpy", seed=cfg.seed)
            writer.add_images("eval_imgs", img_grid, global_step, dataformats="NHWC")
        writer.flush()
    
    # Final test set evaluation
    print("\nRunning final test set evaluation...")
    test_metrics = evaluate_test_set(model, test_dataloader, device, noise_scheduler, cfg.task_name, writer, global_step)
    
    writer.close()


def main():
    ROOT_IMAGE_DIR = get_repo_dir() / Path("data/datasets/unpaired_32x32")
    METADATA_PATH = ROOT_IMAGE_DIR / "metadata.jsonl"

    cfg = TrainConfig_T2C_Glyff(
        image_size=32,
        train_batch_size=32,
        eval_batch_size=16,
        encoder_dim=512,
        save_image_epochs=1,
        save_model_epochs=5,
    )

    # Data loaders with train/val split
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        cfg,
        root_image_dir=ROOT_IMAGE_DIR,
        metadata_path=METADATA_PATH,
        caption_label="Chinese Definition",
        caption_encoder="google-t5/t5-small",
    )

    # Define model
    model = UNet2DConditionModel(
        sample_size=cfg.image_size,             # Target image size
        in_channels=1,                          # Input channels (1 for grayscale)
        out_channels=1,                         # Output channels (1 for grayscale)
        layers_per_block=2,                     # ResNet layers per UNet block
        block_out_channels=(32, 64, 128, 128),  # Output channels for each block
        addition_embed_type="text",             # Condition on text
        cross_attention_dim=cfg.encoder_dim,    # Cross attention dim
        encoder_hid_dim=cfg.encoder_dim,        # Encoder hidden dim size
        encoder_hid_dim_type="text_proj",       # Encoder hidden dim type
        down_block_types=(                      # Downsample block types (Top -> Bottom)
            "DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"
        ),
        up_block_types=(                        # Upsample block types (Bottom -> Top)
            "UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D"
        ),
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Model Parameters: {total_params:,}")

    # Optimizer and schedulers
    optimizer, lr_scheduler = create_optimizer_and_scheduler(model, cfg, train_dataloader)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    inference_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000)

    # Train the model
    print("Starting training...")
    train_loop(cfg, train_dataloader, val_dataloader, test_dataloader, model, optimizer, lr_scheduler, noise_scheduler, inference_scheduler)
    print("Training completed!")


if __name__ == "__main__":
    main()
