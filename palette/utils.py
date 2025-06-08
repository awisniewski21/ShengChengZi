import math
from typing import Tuple

import numpy as np
import torch

from configs import TrainConfig_C2C_Palette


def make_beta_schedule(config: TrainConfig_C2C_Palette, phase: str = "train", cosine_s: float = 8e-3):
    """
    Create a beta schedule for diffusion.
    
    Args:
        config: TrainConfig_C2C_Palette object containing schedule parameters
        phase: Phase to use when extracting from config ('train' or 'test')
        cosine_s: Cosine schedule parameter (only used for cosine schedule)
    """
    # Extract parameters from config
    schedule = config.schedule
    if phase == "train":
        n_timestep = config.n_timestep_train
        linear_start = config.linear_start_train
        linear_end = config.linear_end_train
    else:  # test phase
        n_timestep = config.n_timestep_test
        linear_start = config.linear_start_test
        linear_end = config.linear_end_test
    
    if schedule == "quad":
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
    elif schedule == "linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "warmup10":
        betas = _warmup_beta(n_timestep, linear_start, linear_end, 0.1)
    elif schedule == "warmup50":
        betas = _warmup_beta(n_timestep, linear_start, linear_end, 0.5)
    elif schedule == "const":
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(f"Schedule '{schedule}' is not implemented")
    return betas


def update_model_average(net_cur: torch.nn.Module, net_ema: torch.nn.Module, beta: float):
    """ Exponential moving average (EMA) update for model parameters """
    for p_cur, p_ema in zip(net_cur.parameters(), net_ema.parameters()):
        w_cur, w_ema = p_cur.data, p_ema.data
        if w_ema is None:
            p_ema.data = w_cur
        else:
            p_ema.data = w_ema * beta + (1 - beta) * w_cur


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple = (1, 1, 1, 1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


###
### Helper Functions
###

def _warmup_beta(n_timestep: int, linear_start: float, linear_end: float, warmup_frac: float):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas
