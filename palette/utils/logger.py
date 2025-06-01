from __future__ import annotations

from datetime import datetime
import logging
import os
from typing import Dict

import pandas as pd
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from core.utils.image_utils import tensor_to_image
from configs.char2char.palette import TrainConfig_C2C_Palette


class MetricsTracker:
    """
    Track numerical metrics and compute their averages
    """
    def __init__(self, *keys, phase: str = "train"):
        self.phase = phase
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def update(self, key: str, value, n: int = 1):
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.loc[key, "total"] / self._data.loc[key, "counts"]

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def get_metrics_dict(self):
        return {f"{self.phase}/{k}": v for k, v in dict(self._data["average"]).items()}


class MetricsLogger:
    """ 
    Wrapper around "tensorboard" library to record metrics and visuals
    """
    def __init__(self, config: TrainConfig_C2C_Palette, logger: InfoLogger):
        self.config = config
        self.logger = logger

        self.epoch = 0
        self.iter = 0
        self.phase = "train"

        if config.tensorboard and config.global_rank == 0:
            self.log_dir = self.logger.log_dir
            self.writer = SummaryWriter(self.log_dir)

    def set_iter(self, epoch: int, iter: int, phase: str = "train"):
        self.phase = phase
        self.epoch = epoch
        self.iter = iter

    def save_images(self, results: Dict):
        pass
        # result_path = os.path.join(self.opt["path"]["results"], self.phase, str(self.epoch))
        # os.makedirs(result_path, exist_ok=True)

        # for name, output in zip(results["name"], results["result"]):
        #     Image.fromarray(tensor_to_image(output)).save(os.path.join(result_path, name))

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

    def __getattr__(self, name: str):
        """
        If name is a tensorboard method, wraps the function to add the phase and step information
        If name is not a tensorboard method, performs default getattr() behavior
        """
        if not hasattr(self.writer, name):
            return object.__getattribute__(self, name)
        else:
            fn = getattr(self.writer, name)
            def wrapper(tag, data, *args, **kwargs):
                fn(f"{self.phase}/{tag}", data, self.iter, *args, **kwargs)
            return wrapper


class InfoLogger:
    """
    Wrapper around "logging" library, but only prints information for GPU 0
    """
    def __init__(self, config: TrainConfig_C2C_Palette):
        self.config = config
        self.rank = config.global_rank
        self.phase = config.phase

        self.log_dir = os.path.join(config.log_dir, config.name + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.setup_logger(None, self.log_dir, config.phase, level=logging.INFO, screen=False)
        self.logger = logging.getLogger(config.phase)

    @staticmethod
    def setup_logger(logger_name: str, root: str, phase: str, level: int = logging.INFO, screen: bool = False):
        """ Set up logger """
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

        fmt = logging.Formatter("%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler = logging.FileHandler(os.path.join(root, f"{phase}.log"), mode="a+")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        if screen:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(fmt)
            logger.addHandler(stream_handler)

    def __getattr__(self, name: str):
        """
        If name is a logger method, calls the function on GPU 0 and ignores it on all other GPUs
        If name is not a logger method, performs default getattr() behavior
        """
        if not hasattr(self.logger, name):
            return object.__getattribute__(self, name)
        elif self.rank == 0: # Only call logger methods on GPU 0
            fn = getattr(self.logger, name)
            def wrapper(*args, **kwargs):
                fn(*args, **kwargs)
            return wrapper
        else:
            return lambda *args, **kwargs: None
