import rich_click as click
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel

from configs import TrainConfig_T2C_Glyff
from core.models.t2c_glyffuser import TrainModel_T2C_Glyffuser


def test_t2c_glyffuser(cfg: TrainConfig_T2C_Glyff):
    assert cfg.load_checkpoint_path is not None, "Checkpoint path must be provided for evaluation"

    net = UNet2DConditionModel(
        sample_size=cfg.image_size,
        in_channels=1,
        out_channels=1,
        down_block_types=("DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels=(32, 64, 128, 128),
        layers_per_block=2,
        cross_attention_dim=cfg.encoder_dim,
        encoder_hid_dim=cfg.encoder_dim,
        encoder_hid_dim_type="text_proj",
        addition_embed_type="text",
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    inference_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000)

    model = TrainModel_T2C_Glyffuser(
        config=cfg,
        net=net,
        optimizer=None,
        lr_scheduler=None,
        noise_scheduler=noise_scheduler,
        inference_scheduler=inference_scheduler,
    )
    model.load_checkpoint("test")

    test_metrics = model.test()

    return test_metrics


@click.command()
@click.option("-p", "--load-checkpoint-path", type=str, required=True,             help="Path to the checkpoint file")
@click.option("-r", "--run-name-prefix",      type=str, default="test_t2c_glyff",  help="Run name prefix for logging")
@click.option("-c", "--use-colab",            is_flag=True,                        help="Use Google Colab environment paths")
def main(*args, **kwargs):
    cfg = TrainConfig_T2C_Glyff(*args, **kwargs)
    return test_t2c_glyffuser(cfg)


if __name__ == "__main__":
    main()
