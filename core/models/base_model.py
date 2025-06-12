import pickle
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from core.configs import TrainConfigBase
from core.dataset.datasets import get_dataloaders
from core.utils.train_utils import get_device


class TrainModelBase(ABC):
    """
    Base model class that provides common training functionality
    """

    def __init__(
        self,
        *,
        config: TrainConfigBase,
        net: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ):
        self.config = config
        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.device = get_device()
        self.net = self.net.to(self.device)
        self.run_name = f"{self.config.run_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_epoch = 1
        self.global_step = 0
        self.best_val_loss = float("inf")

        self.train_dataloader, self.val_dataloader, self.test_dataloader = get_dataloaders(
            self.config, 
            root_image_dir=self.config.root_image_dir, 
            metadata_path=self.config.image_metadata_path,
            verbose=self.config.load_checkpoint_path is None,
        )

    def train(self):
        """
        Main training loop that handles training, evaluation, and logging
        """
        print(f"Starting {self.config.run_name_prefix} training from epoch {self.current_epoch} for {self.config.num_epochs - self.current_epoch + 1} more epochs...")

        assert self.optimizer is not None, "Optimizer must be defined for training"
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        for epoch in range(self.current_epoch, self.config.num_epochs + 1):
            self.current_epoch = epoch

            # Training epoch
            train_metrics = self.train_epoch()
            train_metrics = {k: np.mean(v) for k, v in train_metrics.items()} # Average metrics
            self.log_metrics(train_metrics, self.current_epoch, "train")
            print(f"Epoch {epoch}/{self.config.num_epochs}: {train_metrics}")

            # Validation epoch
            val_loss = None
            if self.val_dataloader is not None and epoch % self.config.eval_epoch_interval == 0:
                val_metrics, _ = self.eval_epoch("val")
                val_metrics = {k: np.mean(v) for k, v in val_metrics.items()} # Average metrics
                self.log_metrics(val_metrics, self.current_epoch, "val")
                val_loss = val_metrics["loss"]
                print(f"Validation at epoch {epoch}: {val_metrics}")

            # Save model checkpoint
            self.save_checkpoint(val_loss=val_loss)

        print("Training completed!")
        self.writer.close()


    def train_epoch(self) -> Dict:
        """
        Train the model for one epoch and return metrics
        """
        self.net.train()
        epoch_metrics = defaultdict(list)

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        for batch_data in pbar:
            # Forward pass, loss computation, and backward pass
            loss = self.train_step(batch_data)

            epoch_metrics["loss"].append(loss)
            epoch_metrics["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            self.global_step += 1

            pbar.set_postfix({"loss": f"{np.mean(epoch_metrics['loss']):.4f}"})

            if self.global_step % self.config.log_step_interval == 0:
                self.writer.add_scalar("train_step/loss", loss, self.global_step)
                if self.lr_scheduler is not None:
                    self.writer.add_scalar("train_step/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
                self.writer.flush()

        return epoch_metrics

    @abstractmethod
    def train_step(self, batch_data) -> float:
        """
        Perform a single training step on a batch of data
        Includes the forward pass, loss computation, and backward pass
        """
        pass

    def eval_epoch(self, phase: str) -> Tuple[Dict, List]:
        """
        Evaluate the model for one epoch and return the results
        """
        self.net.eval()
        eval_metrics = defaultdict(list)
        eval_results = defaultdict(list)

        if phase.lower() == "val":
            dataloader = self.val_dataloader
            desc = "Validation"
        elif phase.lower() == "test":
            dataloader = self.test_dataloader
            desc = "Test"
        else:
            raise ValueError(f"Unknown evaluation phase: '{phase}' - must be 'val' or 'test'")

        with torch.no_grad():
            for batch_ix, batch_data in tqdm(enumerate(dataloader), desc=desc):
                loss, out_grid_img, out_labels, metrics, info = self.eval_step(batch_data, phase)
                eval_metrics["loss"].append(loss)
                for key, value in metrics.items():
                    eval_metrics[key].append(value)
                eval_results["out_grid_imgs"].append(out_grid_img)
                eval_results["out_labels"].append(out_labels)
                eval_results["info"].append(info)

                if batch_ix == 0 or phase.lower() == "test":
                    self._log_image_grid(out_grid_img, phase, batch_ix, out_labels=out_labels)

        return eval_metrics, eval_results

    @abstractmethod
    def eval_step(self, batch_data, phase: str) -> Tuple[float, torch.Tensor, List[str] | None, Dict, Dict]:
        """
        Perform a single evaluation step on a batch of data
        Includes forward pass, loss computation, and logging output images
        """
        pass

    def test(self) -> Tuple[Dict, Dict]:
        """
        Run inference on the model using the test dataset and log the results
        """
        print(f"Running {self.config.run_name_prefix} inference on the test set...")

        assert self.test_dataloader is not None, "Test dataloader must be defined for testing"
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        test_metrics, test_results = self.eval_epoch("test")
        self.writer.close()
        with open(self.log_dir / "test_metrics.pkl", "wb") as f:
            pickle.dump(test_metrics, f)
        with open(self.log_dir / "test_results.pkl", "wb") as f:
            pickle.dump(test_results, f)
        return test_metrics, test_results

    def inference(self, input_data) -> Tuple[torch.Tensor, List[str] | None, Dict]:
        """
        Run inference on the model using the provided input data
        """
        print(f"Running {self.config.run_name_prefix} inference on the input data...")

        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.net.eval()
        with torch.no_grad():
            out_grid_img, out_labels, info = self.inference_step(input_data)
            self._log_image_grid(out_grid_img, "inference", 0, out_labels=out_labels)
        return out_grid_img, out_labels, info

    def inference_step(self, input_data) -> Tuple[torch.Tensor, List[str] | None, Dict]:
        """
        Perform a single inference step on input data
        Includes forward pass and logging output images
        """
        pass

    def get_checkpoint_data(self) -> Dict:
        """
        Get data to save in the model checkpoint
        Override this method to include additional attributes for derived models
        """
        return {
            "config": self.config.to_dict(),
            "run_name": self.run_name,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
        }

    def save_checkpoint(self, val_loss: float | None = None):
        """
        Save the current model to a checkpoint file
        """
        new_best_chkpt = val_loss is not None and val_loss < self.best_val_loss
        if new_best_chkpt:
            self.best_val_loss = val_loss

        # Latest checkpoint
        chkpt_data = self.get_checkpoint_data()
        torch.save(chkpt_data, self.checkpoint_dir / f"{self.config.run_name_prefix}_latest.pt")

        # Best checkpoint
        if new_best_chkpt:
            torch.save(chkpt_data, self.checkpoint_dir / f"{self.config.run_name_prefix}_best.pt")
            print(f"Saved new best checkpoint at epoch {self.current_epoch} (best_val_loss: {self.best_val_loss:.4f})")

        # Periodic checkpoint
        chkpt_int = self.config.checkpoint_epoch_interval
        if chkpt_int > 0 and self.current_epoch > 1:
            if self.current_epoch % chkpt_int == 0 or self.current_epoch == self.config.num_epochs:
                torch.save(chkpt_data, self.checkpoint_dir / f"{self.config.run_name_prefix}_epoch_{self.current_epoch}.pt")
                print(f"Saved new checkpoint at epoch {self.current_epoch}")

    def load_checkpoint(self, phase: str):
        """
        Load a model from a checkpoint file
        """
        chkpt_data = torch.load(self.config.load_checkpoint_path, map_location=self.device)
        self.load_checkpoint_data(chkpt_data, phase)
        print(f"Loaded checkpoint from {self.config.load_checkpoint_path}")
        print(f"  Resuming from epoch {self.current_epoch}, step {self.global_step}")

    def load_checkpoint_data(self, chkpt_data: Dict, phase: str):
        """
        Load a model using the checkpoint data
        Override this method to load additional attributes for derived models
        """
        # Load and update config
        chkpt_config: Dict = chkpt_data["config"]
        chkpt_config.update({
            "run_name_prefix": self.config.run_name_prefix,
            "load_checkpoint_path": self.config.load_checkpoint_path,
            "use_colab": self.config.use_colab,
        })
        self.config = self.config.from_dict(chkpt_config)

        # Load and update model state
        self.current_epoch = chkpt_data["current_epoch"] + 1 # Start at next epoch
        self.global_step = chkpt_data["global_step"]
        self.best_val_loss = chkpt_data.get("best_val_loss", float("inf"))
        self.net.load_state_dict(chkpt_data["model_state_dict"])
        if phase.lower() == "train":
            self.optimizer.load_state_dict(chkpt_data["optimizer_state_dict"])
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(chkpt_data["lr_scheduler_state_dict"])

        # Reinitialize dataloaders with updated config
        self.train_dataloader, self.val_dataloader, self.test_dataloader = get_dataloaders(
            self.config,
            root_image_dir=self.config.root_image_dir,
            metadata_path=self.config.image_metadata_path,
        )

    @property
    def log_dir(self) -> Path:
        return self._create_dir(self.config.output_dir / "logs" / self.run_name)

    @property
    def images_dir(self) -> Path:
        return self._create_dir(self.log_dir / "images")

    @property
    def checkpoint_dir(self) -> Path:
        return self._create_dir(self.log_dir / "checkpoints")


    ###
    ### Helper Functions
    ###

    def log_metrics(self, metrics: Dict, log_step: int, phase: str):
        """
        Log metrics to TensorBoard writer
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f"{phase}/{key}", value, log_step)
        self.writer.flush()

    def _log_image_grid(self, out_grid_img: torch.Tensor, phase: str, batch_ix: int, out_labels: List[str] | None = None):
        """
        Log a grid of output images to TensorBoard and save to disk
        """
        self.writer.add_image(f"{phase}/images", out_grid_img, self.current_epoch)
        if out_labels is not None:
            self.writer.add_text(f"{phase}/image_captions", str(dict(enumerate(out_labels))), self.current_epoch)

        grid_img_np = out_grid_img.permute(1, 2, 0).detach().cpu().numpy()
        plt.imsave(self.images_dir / f"{phase}_epoch_{self.current_epoch}_{batch_ix}.png", grid_img_np)
        if "ipykernel" in sys.modules: # Only display images inside Jupyter notebooks
            plt.imshow(grid_img_np)
            plt.show()

    def _create_dir(self, dir_path: Path) -> Path:
        """
        Create a directory if it doesn't exist
        """
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
