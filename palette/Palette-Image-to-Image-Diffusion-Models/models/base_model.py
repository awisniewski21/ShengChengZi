import os
from abc import abstractmethod
from collections import namedtuple
from functools import partial
from typing import Callable, Dict, List

import torch
import torch.nn as nn
from core.logger import InfoLogger, MetricsLogger
from core.utils import set_device
from torch.utils.data import DataLoader

from models.base_network import BaseNetwork

CustomResult = namedtuple("CustomResult", "name result")


class BaseModel:
    def __init__(
        self,
        opt: Dict,
        phase_loader: DataLoader,
        val_loader: DataLoader,
        metrics: List[Callable],
        logger: InfoLogger,
        writer: MetricsLogger,
    ):
        """ Initialize base model attributes """
        self.opt = opt
        self.phase_loader = phase_loader
        self.val_loader = val_loader
        self.metrics = metrics
        self.logger = logger # Logger to log file (only works on GPU 0)
        self.writer = writer # Writer to tensorboard and result file

        self.phase = opt["phase"]
        self.batch_size = self.opt["datasets"][self.phase]["dataloader"]["kwargs"]["batch_size"]
        self.set_device = partial(set_device, rank=opt["global_rank"])
        self.epoch = 0
        self.iter = 0

        self.schedulers = []
        self.optimizers = []
        self.results_dict = CustomResult([], [])

    def train(self) -> None:
        """
        Main training loop
        """
        while self.epoch <= self.opt["train"]["n_epoch"] and self.iter <= self.opt["train"]["n_iter"]:
            self.epoch += 1
            if self.opt["distributed"]:
                # When shuffle=True, this ensures all replicas use a different random ordering for each epoch
                self.phase_loader.sampler.set_epoch(self.epoch)

            train_log = self.train_step()
            train_log.update({"epoch": self.epoch, "iters": self.iter})

            for key, value in train_log.items():
                self.logger.info(f"{str(key):5s}: {value}\t")

            if self.epoch % self.opt["train"]["save_checkpoint_epoch"] == 0:
                self.logger.info(f"Saving the self at the end of epoch {self.epoch:.0f}")
                self.save_everything()

            if self.epoch % self.opt["train"]["val_epoch"] == 0:
                self.logger.info("\n\n\n------------------------------Validation Start------------------------------")
                if self.val_loader is None:
                    self.logger.warning("Validation stop where dataloader is None, Skip it.")
                else:
                    val_log = self.val_step()
                    for key, value in val_log.items():
                        self.logger.info(f"{str(key):5s}: {value}\t")
                self.logger.info("\n------------------------------Validation End------------------------------\n\n")
        self.logger.info("Number of Epochs has reached the limit, End.")

    def test(self) -> None:
        """ Test method (to be implemented in subclass) """
        pass

    @abstractmethod
    def train_step(self) -> Dict:
        """ Training step for the model (to be implemented in subclass) """
        raise NotImplementedError("You must specify how to train your networks.")

    @abstractmethod
    def val_step(self) -> Dict:
        """ Validation step for the model (to be implemented in subclass) """
        raise NotImplementedError("You must specify how to do validation on your networks.")

    def print_network(self, network: BaseNetwork) -> None:
        """ Print network structure, only works on GPU 0 """
        if self.opt["global_rank"] != 0:
            return
        if isinstance(network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            network = network.module

        s = str(network)
        n = sum(p.numel() for p in network.parameters())
        net_struc_str = f"{network.__class__.__name__}"
        self.logger.info(f"Network structure: {net_struc_str}, with parameters: {n:,d}")
        self.logger.info(s)

    def save_network(self, network: BaseNetwork, network_label: str) -> None:
        """ Save network structure, only works on GPU 0 """
        if self.opt["global_rank"] != 0:
            return
        save_filename = f"{self.epoch}_{network_label}.pth"
        save_path = os.path.join(self.opt["path"]["checkpoint"], save_filename)
        if isinstance(network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, network: BaseNetwork, network_label: str, strict: bool = True) -> None:
        """ Load a pretrained network if available """
        if self.opt["path"]["resume_state"] is None:
            return
        self.logger.info(f"Begin loading pretrained model '{network_label}'...")

        model_path = f"{self.opt['path']['resume_state']}_{network_label}.pth"
        if not os.path.exists(model_path):
            self.logger.warning(f"Not loading pretrained model from '{model_path}' since it does not exist.")
            return

        self.logger.info(f"Loading pretrained model from '{model_path}'...")
        if isinstance(network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            network = network.module
        network.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: set_device(storage)), strict=strict)

    def save_training_state(self) -> None:
        """ Saves training state during training, only works on GPU 0 """
        if self.opt["global_rank"] != 0:
            return
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), "optimizers and schedulers must be a list."
        state = {
            "epoch": self.epoch,
            "iter": self.iter,
            "schedulers": [s.state_dict() for s in self.schedulers],
            "optimizers": [o.state_dict() for o in self.optimizers],
        }
        save_filename = f"{self.epoch}.state"
        save_path = os.path.join(self.opt["path"]["checkpoint"], save_filename)
        torch.save(state, save_path)

    def resume_training(self) -> None:
        """ Resume the optimizers and schedulers for training, only works when phase is train or resume training enabled """
        if self.phase != "train" or self.opt["path"]["resume_state"] is None:
            return
        self.logger.info("Begin loading training states")
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), "optimizers and schedulers must be a list."

        state_path = f"{self.opt['path']['resume_state']}.state"
        if not os.path.exists(state_path):
            self.logger.warning(f"Not loading training state from '{state_path}' since it does not exist.")
            return

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

    def load_everything(self):
        """ Load all model components (to be implemented in subclass) """
        pass

    @abstractmethod
    def save_everything(self):
        """ Save all model components (to be implemented in subclass) """
        raise NotImplementedError("You must specify how to save your networks, optimizers and schedulers.")
