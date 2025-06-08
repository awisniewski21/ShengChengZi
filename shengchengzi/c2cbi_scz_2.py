# import torch
# from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, UNet2DModel  # NOQA
# from diffusers.models.autoencoders.vae import Decoder, Encoder
# from torch.nn import L1Loss
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# from core.configs import TrainConfig_C2CBi_SCZ


# def make_1step_sched(device: str):
#     noise_scheduler_1step = DDPMScheduler()
#     noise_scheduler_1step.set_timesteps(1, device=device)
#     noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.to(device)
#     return noise_scheduler_1step

# def my_vae_encoder_fwd(self: Encoder, sample: torch.Tensor):
#     sample = self.conv_in(sample)
#     l_blocks = []
#     # down
#     for down_block in self.down_blocks:
#         l_blocks.append(sample)
#         sample = down_block(sample)
#     # middle
#     sample = self.mid_block(sample)
#     sample = self.conv_norm_out(sample)
#     sample = self.conv_act(sample)
#     sample = self.conv_out(sample)
#     self.current_down_blocks = l_blocks
#     return sample


# def my_vae_decoder_fwd(self: Decoder, sample: torch.Tensor, latent_embeds: torch.Tensor | None = None):
#     sample = self.conv_in(sample)
#     upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
#     # middle
#     sample = self.mid_block(sample, latent_embeds)
#     sample = sample.to(upscale_dtype)
#     if not self.ignore_skip:
#         skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
#         # up
#         for idx, up_block in enumerate(self.up_blocks):
#             skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
#             # add skip
#             sample = sample + skip_in
#             sample = up_block(sample, latent_embeds)
#     else:
#         for idx, up_block in enumerate(self.up_blocks):
#             sample = up_block(sample, latent_embeds)
#     # post-process
#     if latent_embeds is None:
#         sample = self.conv_norm_out(sample)
#     else:
#         sample = self.conv_norm_out(sample, latent_embeds)
#     sample = self.conv_act(sample)
#     sample = self.conv_out(sample)
#     return sample


# class TrainNetwork_C2CBi_SCZ_2(torch.nn.Module):
#     def __init__(
#         self,
#         cfg: TrainConfig_C2CBi_SCZ,
#         device: str,
#     ):
#         super().__init__()
#         self.cfg = cfg

#         # self.sched = DDPMScheduler(num_train_timesteps=1000)
#         # self.sched.set_timesteps(1000, device=device)

#         self.vae = AutoencoderKL(
#             sample_size=cfg.image_size,
#             in_channels=1,
#             out_channels=1,
#             down_block_types=("DownEncoderBlock2D",),
#             up_block_types=("UpDecoderBlock2D",),
#             block_out_channels=(cfg.image_size,),
#             layers_per_block=2,
#             latent_channels=1,
#         )
#         # self.vae.encoder.forward = my_vae_encoder_fwd.__get__(self.vae.encoder, self.vae.encoder.__class__)
#         # self.vae.decoder.forward = my_vae_decoder_fwd.__get__(self.vae.decoder, self.vae.decoder.__class__)

#         # # add the skip connection convs
#         # self.vae.decoder.skip_conv_1 = torch.nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
#         # self.vae.decoder.skip_conv_2 = torch.nn.Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
#         # self.vae.decoder.skip_conv_3 = torch.nn.Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
#         # self.vae.decoder.skip_conv_4 = torch.nn.Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
#         # self.vae.decoder.ignore_skip = False
#         # self.vae.decoder.gamma = 1

#         # torch.nn.init.constant_(self.vae.decoder.skip_conv_1.weight, 1e-5)
#         # torch.nn.init.constant_(self.vae.decoder.skip_conv_2.weight, 1e-5)
#         # torch.nn.init.constant_(self.vae.decoder.skip_conv_3.weight, 1e-5)
#         # torch.nn.init.constant_(self.vae.decoder.skip_conv_4.weight, 1e-5)

#         self.unet = UNet2DModel(    
#             sample_size=cfg.image_size,
#             in_channels=1,
#             out_channels=1,
#             down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
#             up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
#             block_out_channels=(32, 64, 128, 128),
#             layers_per_block=2,
#             # cross_attention_dim=cfg.encoder_dim, # TODO
#             # encoder_hid_dim=cfg.encoder_dim, # TODO
#             # encoder_hid_dim_type="image_proj",
#             class_embed_type="identity",
#             num_class_embeds=2,
#         )

#         # unet.enable_xformers_memory_efficient_attention()
#         # self.unet.to(device)
#         # self.vae.to(device)
#         # self.timesteps = torch.tensor([999], device="cuda").long()

#     def forward(
#         self,
#         src_imgs: torch.Tensor,
#         timesteps: torch.Tensor,
#         tgt_labels: torch.Tensor,
#         noise_scheduler: DDPMScheduler,
#         noise: torch.Tensor,
#     ):
#         src_imgs_embed = self.vae.encode(src_imgs).latent_dist.sample() * self.vae.config.scaling_factor
#         src_imgs_embed_noisy = noise_scheduler.add_noise(src_imgs_embed, noise, timesteps)

#         pred_imgs_latent = self.unet(src_imgs_embed_noisy, timesteps, class_labels=tgt_labels).sample

#         # pred_imgs = noise_scheduler.step(pred_imgs_latent, timesteps, src_imgs_embed, return_dict=True).prev_sample
#         # pred_imgs = pred_imgs.to(pred_imgs_latent.dtype)

#         # self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks

#         pred_imgs = (self.vae.decode(pred_imgs_latent / self.vae.config.scaling_factor).sample).clamp(-1, 1)
#         return pred_imgs

#     # def forward(self, tgt_imgs_noisy: torch.Tensor, timesteps: torch.Tensor, tgt_labels: torch.Tensor, cond_src_imgs: torch.Tensor):
#     #     return self.unet(tgt_imgs_noisy, timesteps, None, class_labels=tgt_labels.unsqueeze(1), added_cond_kwargs=dict(image_embeds=cond_src_imgs.reshape(cond_src_imgs.shape[0], -1)))
