#!/usr/bin/env python3
"""
Palette C2C model with integrated Palette functionality.
"""

import copy
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from tqdm import tqdm

from configs.c2c_palette import TrainConfig_C2C_Palette
from core.models.base_model import TrainModelBase
from palette.models.palette_network import PaletteNetwork
from palette.models.utils import EMA
from palette.utils.device_utils import set_device, set_seed
from palette.utils.logger import InfoLogger, MetricsLogger, MetricsTracker


class TrainModel_C2C_Palette(TrainModelBase):
    """
    Palette C2C model with full Palette functionality integrated.
    """
    
    def __init__(
        self,
        config: TrainConfig_C2C_Palette,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        networks: List[PaletteNetwork],
        losses: List[Callable],
        sample_num: int,
        optimizers: List[torch.optim.Optimizer],
        metrics: List[Callable] = None,
        ema_scheduler: EMA | None = None,
        **kwargs
    ):
        """
        Initialize the Palette C2C model with full functionality.
        
        Args:
            networks: List of Palette networks
            losses: List of loss functions
            sample_num: Number of samples for generation
            optimizers: List of optimizer configurations
            metrics: List of metric functions
            ema_scheduler: EMA scheduler configuration
        """
        # Store Palette-specific components before calling super().__init__
        self.palette_networks = networks
        self.palette_losses = losses
        self.sample_num = sample_num
        self.palette_optimizers = optimizers
        self.palette_metrics = metrics or []
        self.ema_scheduler_config = ema_scheduler
        
        # Get the main network as the model for base class
        main_model = networks[0] if networks else torch.nn.Identity()
        
        # Create dummy optimizer and scheduler (will be replaced in setup)
        dummy_optimizer = torch.optim.Adam([torch.tensor(1.0, requires_grad=True)])
        dummy_scheduler = torch.optim.lr_scheduler.StepLR(dummy_optimizer, step_size=1)
        
        super().__init__(
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            model=main_model,
            optimizer=dummy_optimizer,
            lr_scheduler=dummy_scheduler,
            task_prefix="train_palette_c2c",
            **kwargs
        )
        
        # Setup Palette-specific components
        self.setup_palette_components()

    def setup_palette_components(self):
        """Setup Palette-specific training components."""
        # Setup distributed training if needed
        if self.config.local_rank is None:
            self.config.local_rank = self.config.global_rank = 0

        if self.config.distributed:
            torch.cuda.set_device(int(self.config.local_rank))
            print(f"Using GPU {int(self.config.local_rank)} for training")
            torch.distributed.init_process_group(
                backend="nccl",
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.global_rank,
                group_name="mtorch",
            )

        torch.backends.cudnn.enabled = True
        warnings.warn("Using cuDNN for acceleration (torch.backends.cudnn.enabled=True)")
        set_seed(self.config.seed)

        # Initialize Palette model components
        self.loss_fn = self.palette_losses[0]
        self.netG = self.palette_networks[0]
        
        # Override the model from base class with the actual Palette network
        self.model = self.netG
        
        # Setup EMA if configured
        if self.ema_scheduler_config is not None:
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler_config["ema_decay"])
        else:
            self.netG_EMA = None

        # Setup device handling (override base class device handling)
        self.set_device = partial(set_device, rank=self.config.global_rank)
        
        # Move networks to device
        self.netG = self.set_device(self.netG, distributed=self.config.distributed)
        self.model = self.netG  # Keep model reference in sync
        
        if self.ema_scheduler_config is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.config.distributed)
        
        # Load existing networks if available
        self.load_networks()

        # Setup optimizer (override base class optimizer)
        self.optG = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, self.netG.parameters())), 
            **self.palette_optimizers[0]
        )
        self.optimizer = self.optG  # Keep optimizer reference in sync
        self.optimizers = [self.optG]
        
        # Resume training if needed
        self.resume_training()

        # Set loss and noise schedule on network
        if self.config.distributed:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(device=self.device, phase=self.config.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(device=self.device, phase=self.config.phase)

        # Setup metrics trackers
        self.train_metrics = MetricsTracker(*[m.__name__ for m in self.palette_losses], phase="train")
        self.val_metrics = MetricsTracker(*[m.__name__ for m in self.palette_metrics], phase="val")
        self.test_metrics = MetricsTracker(*[m.__name__ for m in self.palette_metrics], phase="test")

        # Training state (override base class state)
        self.current_epoch = 0
        self.epoch = 0  # Palette uses 'epoch' instead of 'current_epoch'
        self.iter = 0
        self.schedulers = []
        
        # Batch data storage
        self.cond_image = None
        self.gt_image = None
        self.mask = None
        self.mask_image = None
        self.path = None
        self.batch_size = 0
        self.output = None
        self.visuals = None

    # Override base class train method to use Palette's methodology
    def train(self):
        """
        Main training loop using Palette's training methodology.
        """
        print("Starting Palette C2C training...")
        
        while self.epoch <= self.config.num_epochs:
            self.epoch += 1
            self.current_epoch = self.epoch  # Keep base class state in sync
            
            if self.config.distributed:
                # When shuffle=True, this ensures all replicas use a different random ordering for each epoch
                self.train_dataloader.sampler.set_epoch(self.epoch)

            train_log = self.train_epoch()
            train_log.update({"epoch": self.epoch, "iters": self.iter})

            for key, value in train_log.items():
                if hasattr(self, 'logger'):
                    self.logger.info(f"{str(key):5s}: {value}\t")

            if self.epoch % self.config.save_model_epochs == 0:
                if hasattr(self, 'logger'):
                    self.logger.info(f"Saving the model at the end of epoch {self.epoch:.0f}")
                self.save_checkpoint()

            if self.epoch % getattr(self.config, 'val_epoch', self.config.save_image_epochs) == 0:
                if hasattr(self, 'logger'):
                    self.logger.info("\n\n\n------------------------------Validation Start------------------------------")
                if self.val_dataloader is None:
                    if hasattr(self, 'logger'):
                        self.logger.warning("Validation stop where dataloader is None, Skip it.")
                else:
                    val_log = self.validate_epoch()
                    for key, value in val_log.items():
                        if hasattr(self, 'logger'):
                            self.logger.info(f"{str(key):5s}: {value}\t")
                if hasattr(self, 'logger'):
                    self.logger.info("\n------------------------------Validation End------------------------------\n\n")
        
        if hasattr(self, 'logger'):
            self.logger.info("Number of Epochs has reached the limit, End.")
        print("Palette training completed!")

    # Implement BaseTrainingModel abstract methods
    def train_step(self, batch_data: Any) -> float:
        """
        Perform a single training step for Palette model.
        
        Args:
            batch_data: Batch of training data
            
        Returns:
            Loss value for this step
        """
        self.set_input(batch_data)
        self.optG.zero_grad()
        loss = self.netG(self.gt_image, self.cond_image, mask=self.mask)
        loss.backward()
        self.optG.step()

        # Update EMA if configured
        if self.ema_scheduler_config is not None:
            if self.iter > self.ema_scheduler_config["ema_start"] and self.iter % self.ema_scheduler_config["ema_iter"] == 0:
                self.EMA.update_model_average(self.netG_EMA, self.netG)

        return loss.item()

    def validation_step(self, batch_data: Any) -> float:
        """
        Perform a single validation step for Palette model.
        
        Args:
            batch_data: Batch of validation data
            
        Returns:
            Loss value for this step
        """
        with torch.no_grad():
            self.set_input(batch_data)
            if self.config.distributed:
                self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
            else:
                self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)

            # Calculate validation metrics
            total_loss = 0.0
            for met in self.palette_metrics:
                value = met(self.gt_image, self.output)
                total_loss += value
                
            return total_loss / len(self.palette_metrics) if self.palette_metrics else 0.0

    def save_sample_images(self, epoch: int):
        """
        Generate and save sample images for visualization.
        
        Args:
            epoch: Current epoch number
        """
        try:
            # Use a batch from validation set if available
            if self.val_dataloader is not None:
                val_batch = next(iter(self.val_dataloader))
                self.set_input(val_batch)
                
                # Generate samples
                if self.config.distributed:
                    self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)

                # Save images to checkpoint directory
                sample_dir = Path(self.checkpoint_dir) / "samples"
                sample_dir.mkdir(exist_ok=True)
                
                # Get current visuals
                visuals = self.get_current_visuals(phase="val")
                
                # Save each type of visual
                for visual_type, images in visuals.items():
                    for i, img in enumerate(images[:4]):  # Save first 4 samples
                        img_path = sample_dir / f"epoch_{epoch:03d}_{visual_type}_sample_{i}.png"
                        # Convert tensor to PIL and save
                        import torchvision.transforms as transforms
                        to_pil = transforms.ToPILImage()
                        pil_img = to_pil(img.clamp(0, 1))
                        pil_img.save(img_path)
                
                print(f"Saved sample images to {sample_dir}")
                
        except Exception as e:
            print(f"Failed to generate sample images: {e}")

    # Palette-specific helper methods
    def set_input(self, data):
        """Set input tensors to model."""
        if isinstance(data, (tuple, list)) and len(data) == 2:
            # Tuple format from PairedImageDataset: (img_t, img_s)
            # img_t = Traditional characters (condition)
            # img_s = Simplified characters (target) 
            img_src, img_trg = data
            self.cond_image = self.set_device(img_src)  # Traditional as condition
            self.gt_image = self.set_device(img_trg)    # Simplified as target
            self.mask = None  # No mask for char2char task
            self.mask_image = None
            self.path = [f"batch_item_{i}" for i in range(img_src.shape[0])]  # Generate dummy paths
            self.batch_size = img_src.shape[0]
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")

    def get_current_visuals(self, phase: str = "train"):
        """Get current visual results."""
        visuals = {
            "gt_image": (self.gt_image.detach().float().cpu() + 1) / 2,
            "cond_image": (self.cond_image.detach().float().cpu() + 1) / 2,
        }
        if phase != "train":
            visuals.update({
                "output": (self.output.detach().float().cpu() + 1) / 2
            })
        return visuals

    def save_current_results(self):
        """Save current results for logging."""
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append(f"GT_{self.path[idx]}")
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append(f"Process_{self.path[idx]}")
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())

            ret_path.append(f"Out_{self.path[idx]}")
            ret_result.append(self.visuals[idx - self.batch_size].detach().float().cpu())

        return {"name": ret_path, "result": ret_result}

    # Override base class methods to integrate with Palette's training methodology
    def train_epoch(self) -> Dict[str, float]:
        """
        Override base class train_epoch to use Palette's methodology.
        """
        self.netG.train()
        self.train_metrics.reset()
        
        for train_data in tqdm(self.train_dataloader, desc="Train step", disable=self.config.global_rank != 0):
            # Use the train_step method from BaseTrainingModel interface
            loss = self.train_step(train_data)
            
            self.iter += self.batch_size
            self.global_step = self.iter  # Keep base class state in sync
            
            # Update metrics
            self.train_metrics.update(self.loss_fn.__name__, loss)
            
            # Log periodically
            if hasattr(self.config, 'log_iter') and self.iter % self.config.log_iter == 0:
                for key, value in self.train_metrics.get_metrics_dict().items():
                    if hasattr(self, 'logger'):
                        self.logger.info(f"{str(key):5s}: {value}\t")
                    self.writer.add_scalar(key, value)

        for scheduler in self.schedulers:
            scheduler.step()

        return self.train_metrics.get_metrics_dict()

    def validate_epoch(self) -> Dict[str, float]:
        """
        Override base class validate_epoch to use Palette's methodology.
        """
        self.netG.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for val_data in tqdm(self.val_dataloader, desc="Val step", disable=self.config.global_rank != 0):
                # Use validation_step method from BaseTrainingModel interface
                loss = self.validation_step(val_data)
                
                self.iter += self.batch_size
                
                # Update metrics with actual metric functions
                for met in self.palette_metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                    
                # Save visuals
                for key, value in self.get_current_visuals(phase="val").items():
                    self.writer.add_images(key, value)
                    
                self.writer.save_images(self.save_current_results())
                self.writer.flush()

        return self.val_metrics.get_metrics_dict()

    # Override checkpoint methods to be compatible with both base model and Palette
    def save_checkpoint(self):
        """
        Save model checkpoint compatible with both base model and Palette formats.
        """
        # Save in base model format
        super().save_checkpoint()
        
        # Also save in Palette format for compatibility
        self.save_everything()

    def get_model_specific_checkpoint_data(self) -> Dict[str, Any]:
        """
        Get Palette-specific checkpoint data.
        """
        data = {
            "ema_scheduler_config": self.ema_scheduler_config,
            "sample_num": self.sample_num,
            "epoch": self.epoch,
            "iter": self.iter,
        }
        
        if self.ema_scheduler_config is not None and hasattr(self, 'netG_EMA'):
            data["netG_EMA_state_dict"] = self.netG_EMA.state_dict()
            data["EMA_state_dict"] = self.EMA.state_dict() if hasattr(self.EMA, 'state_dict') else None
            
        return data

    def load_model_specific_checkpoint_data(self, checkpoint: Dict[str, Any]):
        """
        Load Palette-specific checkpoint data.
        """
        if "ema_scheduler_config" in checkpoint:
            self.ema_scheduler_config = checkpoint["ema_scheduler_config"]
            
        if "sample_num" in checkpoint:
            self.sample_num = checkpoint["sample_num"]
            
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
            
        if "iter" in checkpoint:
            self.iter = checkpoint["iter"]
            
        if "netG_EMA_state_dict" in checkpoint and hasattr(self, 'netG_EMA'):
            self.netG_EMA.load_state_dict(checkpoint["netG_EMA_state_dict"])
            
        if "EMA_state_dict" in checkpoint and hasattr(self, 'EMA') and checkpoint["EMA_state_dict"] is not None:
            if hasattr(self.EMA, 'load_state_dict'):
                self.EMA.load_state_dict(checkpoint["EMA_state_dict"])

    # Palette-specific training methods (kept for compatibility)

    def test(self):
        """
        Perform testing.
        """
        self.netG.eval()
        self.test_metrics.reset()
        
        with torch.no_grad():
            for phase_data in tqdm(self.train_dataloader, desc="Test step", disable=self.config.global_rank != 0):
                self.set_input(phase_data)
                if self.config.distributed:
                    self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)

                self.iter += self.batch_size
                
                for met in self.palette_metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.test_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                    
                for key, value in self.get_current_visuals(phase="test").items():
                    self.writer.add_images(key, value)
                    
                self.writer.save_images(self.save_current_results())
                self.writer.flush()

        test_log = self.test_metrics.get_metrics_dict()
        test_log.update({"epoch": self.epoch, "iters": self.iter})

        for key, value in test_log.items():
            if hasattr(self, 'logger'):
                self.logger.info(f"{str(key):5s}: {value}\t")

    def load_networks(self):
        """Load pretrained model and training state, only on GPU 0."""
        if self.config.distributed:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler_config is not None:
            self.load_network(network=self.netG_EMA, network_label=f"{netG_label}_ema", strict=False)

    def save_everything(self):
        """Save pretrained model and training state."""
        if self.config.distributed:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler_config is not None:
            self.save_network(network=self.netG_EMA, network_label=f"{netG_label}_ema")
        self.save_training_state()

    def load_network(self, network: nn.Module, network_label: str, strict: bool = True):
        """Load a pretrained network if available."""
        if self.config.resume_state is None:
            return
        if hasattr(self, 'logger'):
            self.logger.info(f"Begin loading pretrained model '{network_label}'...")

        model_path = f"{self.config.resume_state}_{network_label}.pth"
        if not os.path.exists(model_path):
            if hasattr(self, 'logger'):
                self.logger.warning(f"Not loading pretrained model from '{model_path}' since it does not exist.")
            return

        if hasattr(self, 'logger'):
            self.logger.info(f"Loading pretrained model from '{model_path}'...")
        if isinstance(network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            network = network.module
        network.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: self.set_device(storage)), strict=strict)

    def save_network(self, network: nn.Module, network_label: str):
        """Save network structure, only works on GPU 0."""
        if self.config.global_rank != 0:
            return
        save_filename = f"{self.epoch}_{network_label}.pth"
        save_path = os.path.join(self.config.model_dir, self.config.name, save_filename)
        if isinstance(network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def save_training_state(self):
        """Save training state during training, only works on GPU 0."""
        if self.config.global_rank != 0:
            return
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), "optimizers and schedulers must be a list."
        state = {
            "epoch": self.epoch,
            "iter": self.iter,
            "schedulers": [s.state_dict() for s in self.schedulers],
            "optimizers": [o.state_dict() for o in self.optimizers],
        }
        save_filename = f"{self.epoch}.state"
        save_path = os.path.join(self.config.model_dir, self.config.name, save_filename)
        torch.save(state, save_path)

    def resume_training(self):
        """Resume the optimizers and schedulers for training, only works when phase is train or resume training enabled."""
        if self.config.phase != "train" or self.config.resume_state is None:
            return
        if hasattr(self, 'logger'):
            self.logger.info("Begin loading training states")
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), "optimizers and schedulers must be a list."

        state_path = f"{self.config.resume_state}.state"
        if not os.path.exists(state_path):
            if hasattr(self, 'logger'):
                self.logger.warning(f"Not loading training state from '{state_path}' since it does not exist.")
            return

        if hasattr(self, 'logger'):
            self.logger.info(f"Loading training state from '{state_path}'...")
        resume_state = torch.load(state_path, map_location=lambda storage, loc: self.set_device(storage))

        resume_optimizers = resume_state["optimizers"]
        resume_schedulers = resume_state["schedulers"]
        assert len(resume_optimizers) == len(self.optimizers), f"Wrong lengths of optimizers {len(resume_optimizers)} != {len(self.optimizers)}"
        assert len(resume_schedulers) == len(self.schedulers), f"Wrong lengths of schedulers {len(resume_schedulers)} != {len(self.schedulers)}"
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

        self.epoch = resume_state["epoch"]
        self.iter = resume_state["iter"]
