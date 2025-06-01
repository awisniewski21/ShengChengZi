#!/usr/bin/env python3
"""
Random-to-Character (R2C) Glyffuser model trainer.
"""

import torch
from pathlib import Path
from typing import Any, Dict

from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, UNet2DModel
from core.models.base_model import TrainModelBase
from core.utils.eval_utils import DiffusionPipeline_R2C_Glyff
from configs.r2c_glyffuser import TrainConfig_R2C_Glyff


class TrainModel_R2C_Glyffuser(TrainModelBase):
    """
    Random-to-Character Glyffuser model trainer.
    """
    
    def __init__(
        self,
        config: TrainConfig_R2C_Glyff,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        model: UNet2DModel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        noise_scheduler: DDPMScheduler,
        inference_scheduler: DPMSolverMultistepScheduler,
        **kwargs
    ):
        """
        Initialize the Glyffuser R2C model trainer.
        
        Args:
            noise_scheduler: Scheduler for adding noise during training
            inference_scheduler: Scheduler for inference/sampling
        """
        super().__init__(
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            task_prefix="train_glyffuser_r2c",
            **kwargs
        )
        
        self.noise_scheduler = noise_scheduler
        self.inference_scheduler = inference_scheduler

    def train_step(self, batch_data: Any) -> float:
        """
        Perform a single training step for R2C diffusion model.
        
        Args:
            batch_data: Batch of images
            
        Returns:
            Loss value for this step
        """
        b_imgs = batch_data.to(self.device)
        
        # Sample noise and timesteps
        noise = torch.randn_like(b_imgs)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (b_imgs.shape[0],), device=self.device
        ).long()
        
        # Add noise to images
        noisy_images = self.noise_scheduler.add_noise(b_imgs, noise, timesteps)
        
        # Predict noise
        noise_pred = self.model(noisy_images, timesteps, return_dict=False)[0]
        
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
            batch_data: Batch of validation images
            
        Returns:
            Loss value for this step
        """
        b_imgs = batch_data.to(self.device)
        
        # Sample noise and timesteps
        noise = torch.randn_like(b_imgs)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (b_imgs.shape[0],), device=self.device
        ).long()
        
        # Add noise to images
        noisy_images = self.noise_scheduler.add_noise(b_imgs, noise, timesteps)
        
        # Predict noise
        noise_pred = self.model(noisy_images, timesteps, return_dict=False)[0]
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        return loss.item()

    def save_sample_images(self, epoch: int):
        """
        Generate and save sample images using the diffusion pipeline.
        
        Args:
            epoch: Current epoch number
        """
        try:
            # Create diffusion pipeline
            pipeline = DiffusionPipeline_R2C_Glyff(
                unet=self.model,
                scheduler=self.inference_scheduler
            )
            
            # Generate sample images
            sample_images = pipeline(
                batch_size=4,
                generator=torch.Generator(device=self.device).manual_seed(self.config.seed),
                num_inference_steps=50,
            ).images
            
            # Save images
            sample_dir = Path(self.log_dir) / "samples"
            sample_dir.mkdir(exist_ok=True)
            
            for i, img in enumerate(sample_images):
                img_path = sample_dir / f"epoch_{epoch:03d}_sample_{i}.png"
                img.save(img_path)
            
            print(f"Saved {len(sample_images)} sample images to {sample_dir}")
            
        except Exception as e:
            print(f"Failed to generate sample images: {e}")

    def get_model_specific_checkpoint_data(self) -> Dict[str, Any]:
        """
        Get model-specific checkpoint data.
        
        Returns:
            Dictionary containing scheduler states
        """
        return {
            "noise_scheduler_state": self.noise_scheduler.state_dict(),
            "inference_scheduler_state": self.inference_scheduler.state_dict(),
        }

    def load_model_specific_checkpoint_data(self, checkpoint: Dict[str, Any]):
        """
        Load model-specific checkpoint data.
        
        Args:
            checkpoint: Loaded checkpoint dictionary
        """
        if "noise_scheduler_state" in checkpoint:
            self.noise_scheduler.load_state_dict(checkpoint["noise_scheduler_state"])
        if "inference_scheduler_state" in checkpoint:
            self.inference_scheduler.load_state_dict(checkpoint["inference_scheduler_state"])

    def get_evaluation_kwargs(self) -> Dict[str, Any]:
        """
        Get evaluation kwargs specific to R2C model.
        
        Returns:
            Dictionary of evaluation parameters
        """
        return {
            "noise_scheduler": self.noise_scheduler,
            "inference_scheduler": self.inference_scheduler,
        }
