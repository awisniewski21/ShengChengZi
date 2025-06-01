#!/usr/bin/env python3
"""
Character-to-Character model training runner.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from core.dataset.datasets import get_dataloaders
from core.utils.eval_utils import evaluate_char_to_image_grid, evaluate_test_set  # NOQA
from core.utils.repo_utils import get_repo_dir
from core.utils.train_utils import setup_training_environment, create_optimizer_and_scheduler
from configs.c2cbi_scz import TrainConfig_C2CBi_SCZ


def train_loop(
    cfg: TrainConfig_C2CBi_SCZ,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model: UNet2DModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    noise_scheduler: DDPMScheduler,
):
    # Setup training environment
    device, run_name, log_dir, writer = setup_training_environment(cfg, "train_shengchengzi_char2char_bi_orig")
    
    model = model.to(device)

    # Train loop
    global_step = 0
    for epoch in range(cfg.num_epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        total_loss = 0

        model.train()
        for (b_imgs_src, b_imgs_trg, b_labels) in pbar:
            b_imgs_src = b_imgs_src.to(device)
            b_imgs_trg = b_imgs_trg.to(device)
            b_labels = b_labels.to(device)

            timesteps = torch.full((b_imgs_src.shape[0],), 1, device=device).long()

            optimizer.zero_grad()
            pred_img = model(b_imgs_src, timesteps, class_labels=b_labels.unsqueeze(1)).sample
            loss = torch.nn.functional.mse_loss(pred_img.float(), b_imgs_trg.float(), reduction="mean")

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            pbar.set_postfix(**logs)
            writer.add_scalar("Loss/train", logs["loss"], global_step)
            writer.add_scalar("LR", logs["lr"], global_step)
            global_step += 1
            total_loss += logs["loss"]

        writer.add_scalar("Loss/train_epoch", total_loss / len(train_dataloader), epoch)

        model.eval()

        # Validation evaluation
        if len(val_dataloader) > 0:
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_imgs_src, val_imgs_trg, val_labels in val_dataloader:
                    val_imgs_src = val_imgs_src.to(device)
                    val_imgs_trg = val_imgs_trg.to(device)
                    
                    noise = torch.randn_like(val_imgs_trg)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (val_imgs_trg.shape[0],), device=device).long()
                    val_imgs_noisy = noise_scheduler.add_noise(val_imgs_trg, noise, timesteps)
                    noise_pred = model(val_imgs_noisy, timesteps).sample
                    val_loss += torch.nn.functional.mse_loss(noise_pred, noise).item()
                    val_steps += 1
            
            avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
            writer.add_scalar("Loss/validation", avg_val_loss, global_step)
            print(f"Epoch {epoch} - Validation Loss: {avg_val_loss:.4f}")

        # Save model checkpoint
        if epoch % cfg.save_model_epochs == 0 or epoch == cfg.num_epochs - 1:
            if epoch > 0:
                save_dir = Path(cfg.output_dir) / "models" / run_name
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), save_dir / f"{run_name}_epoch_{epoch}.pth")
                torch.save(model.state_dict(), save_dir / f"{run_name}_latest.pth")

        # Evaluate and log images
        if epoch % cfg.save_image_epochs == 0 or epoch == cfg.num_epochs - 1:
            img_grid = evaluate_char_to_image_grid(model, val_dataloader, device)
            writer.add_image("eval_imgs", img_grid, global_step, dataformats="CWH")
        writer.flush()
    
    # Final test set evaluation
    print("\nRunning final test set evaluation...")
    test_metrics = evaluate_test_set(model, test_dataloader, device, noise_scheduler, cfg.task_name, writer, global_step)
    
    writer.close()


def main():
    ROOT_IMAGE_DIR = get_repo_dir() / Path("data/datasets/paired_32x32")
    METADATA_PATH = ROOT_IMAGE_DIR / "metadata.jsonl"

    cfg = TrainConfig_C2CBi_SCZ(
        image_size=32,
        train_batch_size=32,
        eval_batch_size=16,
        encoder_dim=1024,
        save_image_epochs=1,
        save_model_epochs=5,
    )

    # Data loaders with train/val split
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(cfg, root_image_dir=ROOT_IMAGE_DIR, metadata_path=METADATA_PATH)
    # Note: val_dataloader replaces the old eval_dataloader with proper train/val split

    # Model - Using UNet2DModel instead of Char2CharBiModel as in the original notebook
    model = UNet2DModel(    
        sample_size=cfg.image_size,
        in_channels=1,
        out_channels=1,
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels=(32, 64, 128, 128),
        layers_per_block=2,
        class_embed_type="identity",
        num_class_embeds=2,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Model Parameters: {total_params:,}")

    # Optimizer and schedulers
    optimizer, lr_scheduler = create_optimizer_and_scheduler(model, cfg, train_dataloader)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Train the model
    print("Starting training...")
    train_loop(cfg, train_dataloader, val_dataloader, test_dataloader, model, optimizer, lr_scheduler, noise_scheduler)
    print("Training completed!")


if __name__ == "__main__":
    main()
