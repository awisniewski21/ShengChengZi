from functools import partial
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from core.configs import TrainConfig_C2C_Palette
from palette.utils import extract, make_beta_schedule  # NOQA


class PaletteNetwork(nn.Module):
    def __init__(
        self,
        config: TrainConfig_C2C_Palette,
        loss_fn: Callable = torch.nn.functional.mse_loss,
        init_type: str = "kaiming",
        gain: float = 0.02
    ):
        super().__init__()

        self.config = config
        self.loss_fn = loss_fn
        self.init_type = init_type
        self.gain = gain

        if config.module_name == "sr3":
            from .sr3_modules.unet import UNet
            self.denoise_fn = UNet(
                in_channel=config.in_channel,
                out_channel=config.out_channel,
                inner_channel=config.inner_channel,
                channel_mults=tuple(config.channel_mults),
                attn_res=tuple(config.attn_res),
                res_blocks=config.res_blocks,
                dropout=config.dropout,
                image_size=config.image_size
            )
        elif config.module_name == "guided_diffusion":
            from .guided_diffusion_modules.unet import UNet
            self.denoise_fn = UNet(
                image_size=config.image_size,
                in_channel=config.in_channel,
                inner_channel=config.inner_channel,
                out_channel=config.out_channel,
                res_blocks=config.res_blocks,
                attn_res=tuple(config.attn_res),
                dropout=config.dropout,
                channel_mults=tuple(config.channel_mults),
                num_head_channels=config.num_head_channels,
                use_checkpoint=False,
                use_scale_shift_norm=True,
                resblock_updown=True,
                use_new_attention_order=False,
            )
        else:
            raise NotImplementedError(f"Module '{config.module_name}' is not implemented")

    def init_weights(self):
        """
        Initialize network weights
        See: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L68
        """

        def init_func(m: nn.Module):
            classname = m.__class__.__name__
            if classname.find("InstanceNorm2d") != -1:
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
                if self.init_type == "normal":
                    nn.init.normal_(m.weight.data, 0.0, self.gain)
                elif self.init_type == "xavier":
                    nn.init.xavier_normal_(m.weight.data, gain=self.gain)
                elif self.init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif self.init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif self.init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight.data, gain=self.gain)
                elif self.init_type == "none": # Use pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError("initialization method [%s] is not implemented" % self.init_type)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # Propagate to children
        for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(self.init_type, self.gain)

    def set_new_noise_schedule(self, device: torch.device, phase: str):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(config=self.config, phase=phase)
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1.0 - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1.0, gammas[:-1])

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("gammas", to_torch(gammas))
        self.register_buffer("sqrt_recip_gammas", to_torch(np.sqrt(1.0 / gammas)))
        self.register_buffer("sqrt_recipm1_gammas", to_torch(np.sqrt(1.0 / gammas - 1.0)))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - gammas_prev) / (1.0 - gammas)

        # Log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer("posterior_mean_coef1", to_torch(betas * np.sqrt(gammas_prev) / (1.0 - gammas)))
        self.register_buffer("posterior_mean_coef2", to_torch((1.0 - gammas_prev) * np.sqrt(alphas) / (1.0 - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))

        if clip_denoised:
            y_0_hat.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance = self.q_posterior(y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        if noise is None:
            noise = torch.randn_like(y_0)
        return sample_gammas.sqrt() * y_0 + (1.0 - sample_gammas).sqrt() * noise

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond: torch.Tensor, y_t: torch.Tensor | None = None, sample_num: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, "num_timesteps must be greater than sample_num"
        sample_inter = self.num_timesteps // sample_num
        
        if y_t is None:
            y_t = torch.randn_like(y_cond)
        ret_arr = y_t
        for i in tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond)
            y_t_out = ((y_t + 1) / 2).clamp(0, 1)
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t_out], dim=0)
        return y_t_out, ret_arr

    def forward(self, y_0: torch.Tensor, y_cond: torch.Tensor | None = None, noise: torch.Tensor | None = None) -> float:
        # Sample from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,)).long()
        t = t.to(y_0.device)
        gamma_t1 = extract(self.gammas, t - 1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2 - gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        if noise is None:
            noise = torch.randn_like(y_0)
        y_noisy = self.q_sample(y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
        loss = self.loss_fn(noise, noise_hat)

        return loss
