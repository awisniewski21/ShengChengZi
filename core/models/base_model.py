import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs import TrainConfigBase
from core.utils.train_utils import setup_train


class TrainModelBase(ABC):
    """
    Base model class that provides common training functionality
    """

    def __init__(
        self,
        *,
        config: TrainConfigBase,
        train_dataloader: DataLoader | None,
        val_dataloader: DataLoader | None,
        test_dataloader: DataLoader | None,
        net: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        task_prefix: str,
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.task_prefix = task_prefix if optimizer is not None else task_prefix.replace("train", "test")

        self.device, self.run_name, self.log_dir, self.writer = setup_train(self.config, self.task_prefix)

        self.net = self.net.to(self.device)

        self.images_dir = Path(self.log_dir) / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if self.optimizer is not None:
            self.checkpoint_dir = Path(self.log_dir) / "checkpoints"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.current_epoch = 0
        self.global_step = 0


    def train(self):
        """
        Main training loop that handles training, evaluation, and logging
        """
        print(f"Starting {self.task_prefix} training for {self.config.num_epochs} epochs...")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Training epoch
            train_metrics = self.train_epoch()
            self.log_metrics(train_metrics, self.current_epoch, "train")
            print(f"Epoch {epoch}/{self.config.num_epochs}: {train_metrics}")

            # Validation epoch
            if self.val_dataloader is not None and epoch % self.config.eval_epoch_interval == 0:
                val_metrics = self.eval_epoch("val")
                self.log_metrics(val_metrics, self.current_epoch, "val")
                print(f"Validation at epoch {epoch}: {val_metrics}")

            # Save model checkpoint
            if epoch > 0 and epoch % self.config.checkpoint_epoch_interval == 0:
                self.save_checkpoint()
                print(f"Saved checkpoint at epoch {epoch}")

        print("Training completed!")
        self.writer.close()


    def train_epoch(self) -> Dict:
        """
        Train the model for one epoch and return metrics
        """
        self.net.train()
        epoch_metrics = {"loss": 0.0, "count": 0}

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        for batch_data in pbar:
            # Forward pass, loss computation, and backward pass
            loss = self.train_step(batch_data)

            epoch_metrics["loss"] += loss
            epoch_metrics["count"] += 1
            self.global_step += 1

            avg_loss = epoch_metrics["loss"] / epoch_metrics["count"]
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            if self.global_step % self.config.log_step_interval == 0:
                self.writer.add_scalar("train_step/loss", loss, self.global_step)
                if self.lr_scheduler is not None:
                    self.writer.add_scalar("train_step/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
                self.writer.flush()

        return {
            "loss": epoch_metrics["loss"] / epoch_metrics["count"],
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }

    @abstractmethod
    def train_step(self, batch_data) -> float:
        """
        Perform a single training step on a batch of data
        Includes the forward pass, loss computation, and backward pass
        """
        pass

    def eval_epoch(self, phase: str) -> Dict:
        """
        Evaluate the model for one epoch and return metrics
        """
        self.net.eval()
        eval_metrics = {"loss": 0.0, "count": 0}

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
                loss, out_grid_img, out_labels = self.eval_step(batch_data, phase)
                eval_metrics["loss"] += loss
                eval_metrics["count"] += 1

                if batch_ix == 0 or phase.lower() == "test":
                    self._log_image_grid(out_grid_img, phase, batch_ix, out_labels=out_labels)

        return {"loss": eval_metrics["loss"] / eval_metrics["count"]}

    @abstractmethod
    def eval_step(self, batch_data, phase: str) -> Tuple[float, torch.Tensor]:
        """
        Perform a single evaluation step on a batch of data
        Includes forward pass, loss computation, and logging output images
        """
        pass

    def test(self):
        """
        Run inference on the model using the test dataset and log the results
        """
        assert self.test_dataloader is not None, "Test dataloader must be defined for testing"

        print(f"Running {self.task_prefix} inference on the test set...")

        test_metrics = self.eval_epoch("test")
        self.log_metrics(test_metrics, self.current_epoch, "test")
        print(f"Test metrics: {test_metrics}")

        self.writer.close()

    def get_checkpoint_data(self) -> Dict:
        """
        Get data to save in the model checkpoint
        Override this method to include additional attributes for derived models
        """
        return {
            "config": self.config,
            "run_name": self.run_name,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
        }

    def save_checkpoint(self):
        """
        Save the current model to a checkpoint file
        """
        chkpt_data = self.get_checkpoint_data()
        torch.save(chkpt_data, self.checkpoint_dir / f"{self.task_prefix}_latest.pt")
        # if self.current_epoch % (self.config.checkpoint_epoch_interval * 2) == 0:
        #     torch.save(chkpt_data, self.checkpoint_dir / f"{self.task_prefix}_epoch_{self.current_epoch}.pt")

    def load_checkpoint(self, checkpoint_path: str | Path, phase: str):
        """
        Load a model from a checkpoint file
        """
        chkpt_data = torch.load(checkpoint_path, map_location=self.device)
        self.load_checkpoint_data(chkpt_data, phase)
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Resuming from epoch {self.current_epoch}, step {self.global_step}")

    def load_checkpoint_data(self, chkpt_data: Dict, phase: str):
        """
        Load a model using the checkpoint data
        Override this method to load additional attributes for derived models
        """
        self.config = chkpt_data["config"]
        self.current_epoch = chkpt_data["current_epoch"]
        self.global_step = chkpt_data["global_step"]
        self.net.load_state_dict(chkpt_data["model_state_dict"])
        if phase.lower() == "train":
            self.optimizer.load_state_dict(chkpt_data["optimizer_state_dict"])
        if phase.lower() == "train" and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(chkpt_data["lr_scheduler_state_dict"])


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
        if "ipykernel" in sys.modules:
            plt.imshow(grid_img_np)
            plt.show()
