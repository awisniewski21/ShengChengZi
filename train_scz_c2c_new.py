#!/usr/bin/env python3
"""
Character-to-Character (New) model training runner.
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

from core.utils.eval_utils import DiffusionPipelineChar2CharBi
from core.utils.repo_utils import get_repo_dir
from core.dataset.datasets import get_dataloader
from shengchengzi.config.char2char_bi_config import TrainingConfigChar2CharBi
from shengchengzi.models.scz_c2c_bi import Char2CharBiModel


def train_loop(
    cfg: TrainingConfigChar2CharBi,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    model: Char2CharBiModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    noise_scheduler: DDPMScheduler,
):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = model.to(device)

    # Tensorboard logging
    if cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)
    run_name = f"train_shengchengzi_char2char_bi_new_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = str(Path(cfg.output_dir) / "logs" / run_name)
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0
    for epoch in range(cfg.num_epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")

        model.train()
        for (b_imgs_src, b_imgs_trg, b_labels) in pbar:
            b_imgs_src = b_imgs_src.to(device)
            b_imgs_trg = b_imgs_trg.to(device)
            b_labels = b_labels.to(device)

            # --- 1. VAE forward & reconstruction loss ---
            vae_pred_imgs = model.vae(b_imgs_trg).sample
            vae_recon_loss = torch.nn.functional.mse_loss(vae_pred_imgs, b_imgs_trg)

            # --- 2. Diffusion loss in latent space ---
            with torch.no_grad():
                tgt_latent = model.vae.encode(b_imgs_trg).latent_dist.sample() * model.vae.config.scaling_factor
            noise = torch.randn_like(tgt_latent)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b_imgs_trg.shape[0],), device=device).long()
            noisy_tgt_latents = noise_scheduler.add_noise(tgt_latent, noise, timesteps)

            noise_pred = model(noisy_tgt_latents, timesteps, b_imgs_src, b_labels.unsqueeze(1))
            diff_loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # --- Total loss ---
            total_loss = vae_recon_loss + diff_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            pbar.set_postfix(loss=total_loss.detach().item(), lr=lr_scheduler.get_last_lr()[0], step=global_step)
            writer.add_scalar("loss/total_train", total_loss.detach().item(), global_step)
            writer.add_scalar("loss/vae_recon_train", vae_recon_loss.detach().item(), global_step)
            writer.add_scalar("loss/diff_train", diff_loss.detach().item(), global_step)
            writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], global_step)
            global_step += 1

        model.eval()
        pipeline = DiffusionPipelineChar2CharBi(model, noise_scheduler)
        pipeline.set_progress_bar_config(desc="Generating evaluation image grid...")

        # Save model checkpoint
        if epoch % cfg.save_model_epochs == 0 or epoch == cfg.num_epochs - 1:
            if epoch > 0:
                pipeline.save_pretrained(str(Path(cfg.output_dir) / "models" / run_name / f"epoch_{epoch}"))
                pipeline.save_pretrained(str(Path(cfg.output_dir) / "models" / run_name / f"latest"))

        # Evaluate and log images
        if epoch % cfg.save_image_epochs == 0 or epoch == cfg.num_epochs - 1:
            for (b_imgs_src, b_imgs_trg, b_labels) in eval_dataloader:
                b_imgs_src = b_imgs_src.to(device)
                b_labels = b_labels.to(device)
                img_grid = pipeline.evaluate_char_to_image_grid(b_imgs_src, b_labels, batch_size=cfg.eval_batch_size, output_type="numpy", seed=cfg.seed)
                writer.add_images("eval_imgs", img_grid, global_step, dataformats="NHWC")
                break
        writer.flush()
    writer.close()


def main():
    ROOT_IMAGE_DIR = get_repo_dir() / Path("data/datasets/paired_32x32")
    METADATA_PATH = ROOT_IMAGE_DIR / "metadata.jsonl"

    cfg = TrainingConfigChar2CharBi(
        image_size=32,
        train_batch_size=32,
        eval_batch_size=16,
        encoder_dim=1024,
        save_image_epochs=1,
        save_model_epochs=5,
    )

    # Data loaders
    train_dataloader = get_dataloader(cfg, root_image_dir=ROOT_IMAGE_DIR, metadata_path=METADATA_PATH)
    eval_dataloader = get_dataloader(cfg, root_image_dir=ROOT_IMAGE_DIR, metadata_path=METADATA_PATH, batch_size=cfg.eval_batch_size, shuffle=False)

    # Model
    model = Char2CharBiModel(cfg)
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

    # Train the model
    print("Starting training...")
    train_loop(cfg, train_dataloader, eval_dataloader, model, optimizer, lr_scheduler, noise_scheduler)
    print("Training completed!")


if __name__ == "__main__":
    main()
