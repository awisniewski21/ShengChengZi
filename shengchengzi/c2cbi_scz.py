# import torch
# from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

# from configs import TrainConfig_C2CBi_SCZ


# class TrainNetwork_C2CBi_SCZ(torch.nn.Module):
#     def __init__(self, cfg: TrainConfig_C2CBi_SCZ):
#         super().__init__()
#         self.cfg = cfg

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

#         self.unet = UNet2DConditionModel(
#             sample_size=cfg.image_size,
#             in_channels=1,
#             out_channels=1,
#             down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
#             up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
#             block_out_channels=(32, 64, 128, 128),
#             layers_per_block=2,
#             class_embed_type="identity",
#             num_class_embeds=2,
#             cross_attention_dim=cfg.image_size,  # or another suitable value
#         )

#     def forward(
#         self,
#         noisy_tgt_latents: torch.Tensor,
#         timesteps: torch.Tensor,
#         src_imgs: torch.Tensor,
#         tgt_labels: torch.Tensor,
#     ):
#         # Encode source image to latent space
#         src_latent = self.vae.encode(src_imgs).latent_dist.sample() * self.vae.config.scaling_factor
#         encoder_hidden_states = src_latent.squeeze(1)

#         # Condition on source latent using encoder_hidden_states
#         noise_pred = self.unet(noisy_tgt_latents, timesteps, encoder_hidden_states, class_labels=tgt_labels).sample

#         return noise_pred
