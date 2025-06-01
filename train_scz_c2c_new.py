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

from core.dataset.datasets import get_dataloaders
from core.utils.eval_utils import DiffusionPipelineChar2CharBi, evaluate_test_set  # NOQA
from core.utils.repo_utils import get_repo_dir
from core.utils.train_utils import setup_training_environment, create_optimizer_and_scheduler
from shengchengzi.config.char2char_bi_config import TrainingConfigChar2CharBi
from shengchengzi.models.scz_c2c_bi import Char2CharBiModel


def train_loop(
    cfg: TrainingConfigChar2CharBi,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model: Char2CharBiModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    noise_scheduler: DDPMScheduler,
):
    # Setup training environment
    device, run_name, log_dir, writer = setup_training_environment(cfg, "train_shengchengzi_char2char_bi_new")
    
    model = model.to(device)

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

        # Validation evaluation
        if len(val_dataloader) > 0:
            val_total_loss = 0.0
            val_vae_loss = 0.0
            val_diff_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_imgs_src, val_imgs_trg, val_labels in val_dataloader:
                    val_imgs_src = val_imgs_src.to(device)
                    val_imgs_trg = val_imgs_trg.to(device)
                    val_labels = val_labels.to(device)
                    
                    val_vae_recon_loss, val_diff_loss_val = model(val_imgs_src, val_imgs_trg, val_labels)
                    val_total_loss_val = val_vae_recon_loss + val_diff_loss_val
                    
                    val_total_loss += val_total_loss_val.item()
                    val_vae_loss += val_vae_recon_loss.item()
                    val_diff_loss += val_diff_loss_val.item()
                    val_steps += 1
            
            if val_steps > 0:
                avg_val_total_loss = val_total_loss / val_steps
                avg_val_vae_loss = val_vae_loss / val_steps
                avg_val_diff_loss = val_diff_loss / val_steps
                
                writer.add_scalar("loss/total_validation", avg_val_total_loss, global_step)
                writer.add_scalar("loss/vae_recon_validation", avg_val_vae_loss, global_step)
                writer.add_scalar("loss/diff_validation", avg_val_diff_loss, global_step)
                print(f"Epoch {epoch} - Val Total Loss: {avg_val_total_loss:.4f}, Val VAE Loss: {avg_val_vae_loss:.4f}, Val Diff Loss: {avg_val_diff_loss:.4f}")

        # Save model checkpoint
        if epoch % cfg.save_model_epochs == 0 or epoch == cfg.num_epochs - 1:
            if epoch > 0:
                save_dir = Path(cfg.output_dir) / "models" / run_name
                save_dir.mkdir(parents=True, exist_ok=True)
                pipeline.save_pretrained(str(save_dir / f"epoch_{epoch}"))
                pipeline.save_pretrained(str(save_dir / "latest"))

        # Evaluate and log images
        if epoch % cfg.save_image_epochs == 0 or epoch == cfg.num_epochs - 1:
            for (b_imgs_src, b_imgs_trg, b_labels) in val_dataloader:
                b_imgs_src = b_imgs_src.to(device)
                b_labels = b_labels.to(device)
                img_grid = pipeline.evaluate_char_to_image_grid(b_imgs_src, b_labels, batch_size=cfg.eval_batch_size, output_type="numpy", seed=cfg.seed)
                writer.add_images("eval_imgs", img_grid, global_step, dataformats="NHWC")
                break
        writer.flush()
    
    # Final test set evaluation
    print("\nRunning final test set evaluation...")
    # Note: For this model, we'll evaluate using the VAE+diffusion loss
    if test_dataloader is not None and len(test_dataloader) > 0:
        test_total_loss = 0.0
        test_steps = 0
        with torch.no_grad():
            for test_imgs_src, test_imgs_trg, test_labels in test_dataloader:
                test_imgs_src = test_imgs_src.to(device)
                test_imgs_trg = test_imgs_trg.to(device)
                test_labels = test_labels.to(device)
                
                test_vae_recon_loss, test_diff_loss = model(test_imgs_src, test_imgs_trg, test_labels)
                test_total_loss += (test_vae_recon_loss + test_diff_loss).item()
                test_steps += 1
        
        avg_test_loss = test_total_loss / test_steps if test_steps > 0 else 0.0
        writer.add_scalar("loss/total_test", avg_test_loss, global_step)
        print(f"Test Results - Total Loss: {avg_test_loss:.4f}, Samples: {test_steps * test_dataloader.batch_size}")
    
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

    # Data loaders with train/val split
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(cfg, root_image_dir=ROOT_IMAGE_DIR, metadata_path=METADATA_PATH)
    # Note: val_dataloader replaces the old eval_dataloader with proper train/val split

    # Model
    model = Char2CharBiModel(cfg)
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
