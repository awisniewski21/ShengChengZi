#!/usr/bin/env python3
"""
Base model class for all training runners in the ShengChengZi project.
Provides common functionality like training loops, evaluation, logging, and checkpoint management.
"""

import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from configs.base_config import TrainConfigBase
from core.utils.eval_utils import evaluate_test_set
from core.utils.train_utils import setup_training_environment


class TrainModelBase(ABC):
    """
    Base model class that provides common training functionality.
    All specific model trainers should inherit from this class.
    """
    
    def __init__(
        self,
        config: TrainConfigBase,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        task_prefix: str,
        **kwargs
    ):
        """
        Initialize the base training model.
        
        Args:
            config: Training configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader  
            test_dataloader: Test data loader
            model: The neural network model to train
            optimizer: Optimizer for training
            lr_scheduler: Learning rate scheduler
            task_prefix: Prefix for logging and checkpoints (e.g., "train_glyffuser_r2c")
            **kwargs: Additional arguments for specific models
        """
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.task_prefix = task_prefix
        
        # Setup training environment
        self.device, self.run_name, self.log_dir, self.writer = setup_training_environment(
            config, task_prefix
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Store additional kwargs for specific models
        self.kwargs = kwargs
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(self.log_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Training setup complete:")
        print(f"  Device: {self.device}")
        print(f"  Run name: {self.run_name}")
        print(f"  Log directory: {self.log_dir}")
        print(f"  Checkpoint directory: {self.checkpoint_dir}")

    def train(self):
        """
        Main training loop that handles epochs, validation, and checkpointing.
        """
        print(f"Starting training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training step
            train_metrics = self.train_epoch()
            
            # Log training metrics
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, epoch)
            
            print(f"Epoch {epoch}/{self.config.num_epochs}: {train_metrics}")
            
            # Validation step
            if self.val_dataloader is not None and (epoch + 1) % self.config.save_image_epochs == 0:
                val_metrics = self.validate_epoch()
                
                # Log validation metrics
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f"val/{key}", value, epoch)
                
                print(f"Validation at epoch {epoch}: {val_metrics}")
            
            # Save model checkpoint
            if (epoch + 1) % self.config.save_model_epochs == 0:
                self.save_checkpoint()
                print(f"Saved checkpoint at epoch {epoch}")
            
            # Generate and save sample images
            if (epoch + 1) % self.config.save_image_epochs == 0:
                self.save_sample_images(epoch)
        
        # Final evaluation on test set
        if self.test_dataloader is not None:
            print("Running final evaluation on test set...")
            test_metrics = self.evaluate_test_set()
            print(f"Final test metrics: {test_metrics}")
            
            # Log test metrics
            for key, value in test_metrics.items():
                self.writer.add_scalar(f"test/{key}", value, self.config.num_epochs)
        
        print("Training completed!")
        self.writer.close()

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch and return metrics.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_metrics = {"loss": 0.0, "count": 0}
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_data in pbar:
            # Forward pass, loss computation, and backward pass
            loss = self.train_step(batch_data)
            
            # Update metrics
            epoch_metrics["loss"] += loss
            epoch_metrics["count"] += 1
            self.global_step += 1
            
            # Update progress bar
            avg_loss = epoch_metrics["loss"] / epoch_metrics["count"]
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            # Log step-level metrics
            if self.global_step % 100 == 0:
                self.writer.add_scalar("train_step/loss", loss, self.global_step)
                self.writer.add_scalar("train_step/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
        
        # Calculate final epoch metrics
        final_metrics = {
            "loss": epoch_metrics["loss"] / epoch_metrics["count"],
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }
        
        return final_metrics

    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch and return metrics.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_metrics = {"loss": 0.0, "count": 0}
        
        with torch.no_grad():
            for batch_data in tqdm(self.val_dataloader, desc="Validation"):
                loss = self.validation_step(batch_data)
                val_metrics["loss"] += loss
                val_metrics["count"] += 1
        
        return {"loss": val_metrics["loss"] / val_metrics["count"]}

    def save_checkpoint(self):
        """
        Save model checkpoint, optimizer state, and training metadata.
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "config": self.config,
            "run_name": self.run_name,
        }
        
        # Add model-specific state if available
        model_specific_state = self.get_model_specific_checkpoint_data()
        if model_specific_state:
            checkpoint.update(model_specific_state)
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest checkpoint
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """
        Load model checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.run_name = checkpoint["run_name"]
        
        # Load model-specific state if available
        self.load_model_specific_checkpoint_data(checkpoint)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Resuming from epoch {self.current_epoch}, step {self.global_step}")

    def evaluate_test_set(self) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Returns:
            Dictionary of test metrics
        """
        try:
            return evaluate_test_set(
                model=self.model,
                test_dataloader=self.test_dataloader,
                device=self.device,
                **self.get_evaluation_kwargs()
            )
        except Exception as e:
            print(f"Test evaluation failed: {e}")
            return {"test_error": 1.0}

    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    def train_step(self, batch_data: Any) -> float:
        """
        Perform a single training step.
        
        Args:
            batch_data: Batch of training data
            
        Returns:
            Loss value for this step
        """
        pass

    @abstractmethod
    def validation_step(self, batch_data: Any) -> float:
        """
        Perform a single validation step.
        
        Args:
            batch_data: Batch of validation data
            
        Returns:
            Loss value for this step
        """
        pass

    @abstractmethod
    def save_sample_images(self, epoch: int):
        """
        Generate and save sample images for visualization.
        
        Args:
            epoch: Current epoch number
        """
        pass

    # Optional methods that can be overridden by subclasses
    
    def get_model_specific_checkpoint_data(self) -> Dict[str, Any]:
        """
        Get model-specific data to include in checkpoints.
        Override this method to save additional model-specific state.
        
        Returns:
            Dictionary of additional checkpoint data
        """
        return {}

    def load_model_specific_checkpoint_data(self, checkpoint: Dict[str, Any]):
        """
        Load model-specific data from checkpoints.
        Override this method to restore additional model-specific state.
        
        Args:
            checkpoint: Loaded checkpoint dictionary
        """
        pass

    def get_evaluation_kwargs(self) -> Dict[str, Any]:
        """
        Get additional keyword arguments for test evaluation.
        Override this method to provide model-specific evaluation parameters.
        
        Returns:
            Dictionary of keyword arguments for evaluation
        """
        return {}
