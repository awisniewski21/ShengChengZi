import copy
from typing import Callable, List

import torch
from tqdm import tqdm

from palette.models.base_model import BaseModel
from palette.models.palette_network import PaletteNetwork
from palette.models.utils import EMA
from palette.utils.logger import MetricsTracker
from configs.c2c_palette import TrainConfig_C2C_Palette


class PaletteModel(BaseModel):
    def __init__(
        self,
        networks: List[PaletteNetwork],
        losses: List[Callable],
        sample_num: int,
        optimizers: List[torch.optim.Optimizer],
        ema_scheduler: EMA | None = None,
        **kwargs,
    ):
        """
        Initialize model
        """
        super(PaletteModel, self).__init__(**kwargs)

        self.loss_fn = losses[0]
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler["ema_decay"])
        else:
            self.ema_scheduler = None

        # Must convert network(s) using self.set_device if using multiple GPUs
        self.netG = self.set_device(self.netG, distributed=self.config.distributed)
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.config.distributed)
        self.load_networks()

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optimizers.append(self.optG)
        self.resume_training()

        if self.config.distributed:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)

        # Metrics trackers
        self.train_metrics = MetricsTracker(*[m.__name__ for m in losses], phase="train")
        self.val_metrics = MetricsTracker(*[m.__name__ for m in self.metrics], phase="val")
        self.test_metrics = MetricsTracker(*[m.__name__ for m in self.metrics], phase="test")

        self.sample_num = sample_num

    def set_input(self, data):
        """ Set input tensors to model """
        self.cond_image = self.set_device(data.get("cond_image"))
        self.gt_image = self.set_device(data.get("gt_image"))
        self.mask = self.set_device(data.get("mask"))
        self.mask_image = data.get("mask_image")
        self.path = data["path"]
        self.batch_size = len(data["path"])

    def get_current_visuals(self, phase: str = "train"):
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
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append(f"GT_{self.path[idx]}")
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append(f"Process_{self.path[idx]}")
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())

            ret_path.append(f"Out_{self.path[idx]}")
            ret_result.append(self.visuals[idx - self.batch_size].detach().float().cpu())

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        for train_data in tqdm(self.phase_loader, desc="Train step", disable=self.config.global_rank != 0):
            self.set_input(train_data)
            self.optG.zero_grad()
            loss = self.netG(self.gt_image, self.cond_image, mask=self.mask)
            loss.backward()
            self.optG.step()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase="train")
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if self.iter % self.config.log_iter == 0:
                for key, value in self.train_metrics.get_metrics_dict().items():
                    self.logger.info(f"{str(key):5s}: {value}\t")
                    self.writer.add_scalar(key, value)
                # for key, value in self.get_current_visuals().items():
                #     self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler["ema_start"] and self.iter % self.ema_scheduler["ema_iter"] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()

        return self.train_metrics.get_metrics_dict()

    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm(self.val_loader, desc="Val step", disable=self.config.global_rank != 0):
                self.set_input(val_data)
                if self.config.distributed:
                    self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase="val")

                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase="val").items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())
                self.writer.flush()

        return self.val_metrics.get_metrics_dict()

    def test(self):
        self.netG.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for phase_data in tqdm(self.phase_loader, desc="Test step", disable=self.config.global_rank != 0):
                self.set_input(phase_data)
                if self.config.distributed:
                    self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase="test")
                for met in self.metrics:
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
            self.logger.info(f"{str(key):5s}: {value}\t")

    def load_networks(self):
        """ Load pretrained model and training state, only on GPU 0 """
        if self.config.distributed:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=f"{netG_label}_ema", strict=False)

    def save_everything(self):
        """ Save pretrained model and training state """
        if self.config.distributed:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=f"{netG_label}_ema")
        self.save_training_state()
