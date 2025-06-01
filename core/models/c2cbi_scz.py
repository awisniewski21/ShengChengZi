#!/usr/bin/env python3
"""
Character-to-Character (C2C) ShengChengZi model trainer.
"""

import torch
from pathlib import Path
from typing import Any, Dict

from diffusers import DDPMScheduler, UNet2DModel
from core.models.base_model import TrainModelBase
from core.utils.eval_utils import evaluate_char_to_image_grid
from configs.c2cbi_scz import TrainConfig_C2CBi_SCZ


class TrainModel_C2C_SCZ(TrainModelBase):
    """
    Character-to-Character ShengChengZi model trainer.
    """
    
    def __init__(
        self,
        config: TrainConfig_C2CBi_SCZ,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        model: UNet2DModel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        noise_scheduler: DDPMScheduler,
        **kwargs
    ):
        """
        Initialize the ShengChengZi C2C model trainer.
        
        Args:
            noise_scheduler: Scheduler for adding noise during training
        """
        super().__init__(
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            task_prefix="train_shengchengzi_c2c",
            **kwargs
        )
        
        self.noise_scheduler = noise_scheduler

    def train_step(self, batch_data: Any) -> float:
        """
        Perform a single training step for C2C diffusion model.
        
        Args:
            batch_data: Tuple of (source_images, target_images, labels)
            
        Returns:
            Loss value for this step
        """
        b_imgs_src, b_imgs_trg, b_labels = batch_data
        b_imgs_src = b_imgs_src.to(self.device)
        b_imgs_trg = b_imgs_trg.to(self.device)
        b_labels = b_labels.to(self.device)
        
        # Sample noise and timesteps
        noise = torch.randn_like(b_imgs_trg)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (b_imgs_trg.shape[0],), device=self.device
        ).long()
        
        # Add noise to target images
        noisy_images = self.noise_scheduler.add_noise(b_imgs_trg, noise, timesteps)
        
        # Concatenate source and noisy target images as input
        model_input = torch.cat([b_imgs_src, noisy_images], dim=1)
        
        # Predict noise
        noise_pred = self.model(model_input, timesteps, return_dict=False)[0]
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        
        return loss.item()

    def validation_step(self, batch_data: Any) -> float:
        """
        Perform a single validation step.
        
        Args:
            batch_data: Tuple of (source_images, target_images, labels)
            
        Returns:
            Loss value for this step
        """
        b_imgs_src, b_imgs_trg, b_labels = batch_data
        b_imgs_src = b_imgs_src.to(self.device)
        b_imgs_trg = b_imgs_trg.to(self.device)
        b_labels = b_labels.to(self.device)
        
        # Sample noise and timesteps
        noise = torch.randn_like(b_imgs_trg)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (b_imgs_trg.shape[0],), device=self.device
        ).long()
        
        # Add noise to target images
        noisy_images = self.noise_scheduler.add_noise(b_imgs_trg, noise, timesteps)
        
        # Concatenate source and noisy target images as input
        model_input = torch.cat([b_imgs_src, noisy_images], dim=1)
        
        # Predict noise
        noise_pred = self.model(model_input, timesteps, return_dict=False)[0]
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        return loss.item()

    def save_sample_images(self, epoch: int):
        """
        Generate and save sample images by performing character translation.
        
        Args:
            epoch: Current epoch number
        """
        try:
            # Get a batch from validation set for sampling
            if self.val_dataloader is not None:
                val_batch = next(iter(self.val_dataloader))
                b_imgs_src, b_imgs_trg, b_labels = val_batch
                b_imgs_src = b_imgs_src[:4].to(self.device)  # Take first 4 samples
                
                # Generate images using reverse diffusion
                sample_images = self._generate_c2c_samples(b_imgs_src)
                
                # Save images
                sample_dir = Path(self.log_dir) / "samples"
                sample_dir.mkdir(exist_ok=True)
                
                # Save a grid showing source -> generated
                grid_path = sample_dir / f"epoch_{epoch:03d}_c2c_grid.png"
                evaluate_char_to_image_grid(
                    source_images=b_imgs_src.cpu(),
                    generated_images=sample_images.cpu(),
                    save_path=str(grid_path)
                )
                
                print(f"Saved character-to-character grid to {grid_path}")
            
        except Exception as e:
            print(f"Failed to generate sample images: {e}")

    def _generate_c2c_samples(self, source_images: torch.Tensor) -> torch.Tensor:
        """
        Generate character-to-character samples using reverse diffusion.
        
        Args:
            source_images: Source character images
            
        Returns:
            Generated target character images
        """
        self.model.eval()
        
        with torch.no_grad():
            # Start with random noise
            generated = torch.randn(
                source_images.shape[0], 1, source_images.shape[2], source_images.shape[3],
                device=self.device
            )
            
            # Reverse diffusion process
            for t in reversed(range(self.noise_scheduler.config.num_train_timesteps)):
                timestep = torch.full((source_images.shape[0],), t, device=self.device).long()
                
                # Concatenate source and current generated as input
                model_input = torch.cat([source_images, generated], dim=1)
                
                # Predict noise
                noise_pred = self.model(model_input, timestep, return_dict=False)[0]
                
                # Remove predicted noise
                generated = self.noise_scheduler.step(noise_pred, t, generated).prev_sample
        
        self.model.train()
        return generated

    def get_model_specific_checkpoint_data(self) -> Dict[str, Any]:
        """
        Get model-specific checkpoint data.
        
        Returns:
            Dictionary containing scheduler state
        """
        return {
            "noise_scheduler_state": self.noise_scheduler.state_dict(),
        }

    def load_model_specific_checkpoint_data(self, checkpoint: Dict[str, Any]):
        """
        Load model-specific checkpoint data.
        
        Args:
            checkpoint: Loaded checkpoint dictionary
        """
        if "noise_scheduler_state" in checkpoint:
            self.noise_scheduler.load_state_dict(checkpoint["noise_scheduler_state"])

    def get_evaluation_kwargs(self) -> Dict[str, Any]:
        """
        Get evaluation kwargs specific to C2C model.
        
        Returns:
            Dictionary of evaluation parameters
        """
        return {
            "noise_scheduler": self.noise_scheduler,
        }
