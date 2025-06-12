import rich_click as click
from diffusers import DDPMScheduler, UNet2DModel

from core.configs import TrainConfig_R2C_Glyff
from core.models.r2c_glyffuser import TrainModel_R2C_Glyffuser


def inference_r2c_glyffuser(cfg: TrainConfig_R2C_Glyff):
    assert cfg.load_checkpoint_path is not None, "Checkpoint path must be provided for inference"

    net = UNet2DModel(
        sample_size=cfg.image_size,
        in_channels=1,
        out_channels=1,
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels=(32, 64, 128, 128),
        layers_per_block=2,
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    inference_scheduler = DDPMScheduler()

    model = TrainModel_R2C_Glyffuser(
        config=cfg,
        net=net,
        optimizer=None,
        lr_scheduler=None,
        noise_scheduler=noise_scheduler,
        inference_scheduler=inference_scheduler,
    )
    model.load_checkpoint("inference")

    # R2C model generates from random noise, so no input data is needed
    model.inference_step()


@click.command()
@click.option("-p", "--load-checkpoint-path", type=str, required=True,                 help="Path to the checkpoint file")
@click.option("-r", "--run-name-prefix",      type=str, default="inference_r2c_glyff", help="Run name prefix for logging")
@click.option("-c", "--use-colab",            is_flag=True,                            help="Use Google Colab environment paths")
def main(*args, **kwargs):
    cfg = TrainConfig_R2C_Glyff(*args, **kwargs)
    return inference_r2c_glyffuser(cfg)


if __name__ == "__main__":
    main()
