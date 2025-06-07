import rich_click as click
from diffusers import DDPMScheduler, UNet2DModel

from configs import TrainConfig_C2CBi_SCZ
from core.models.c2cbi_scz import TrainModel_C2CBi_SCZ


def test_c2cbi_scz(cfg: TrainConfig_C2CBi_SCZ):
    assert cfg.load_checkpoint_path is not None, "Checkpoint path must be provided for evaluation"

    net = UNet2DModel(    
        sample_size=cfg.image_size,
        in_channels=1,
        out_channels=1,
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels=(32, 64, 128, 128),
        layers_per_block=2,
        class_embed_type="identity",
        num_class_embeds=2,
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    model = TrainModel_C2CBi_SCZ(
        config=cfg,
        net=net,
        optimizer=None,
        lr_scheduler=None,
        noise_scheduler=noise_scheduler,
    )
    model.load_checkpoint("test")

    test_metrics = model.test()

    return test_metrics


@click.command()
@click.option("-p", "--load-checkpoint-path", type=str, required=True,           help="Path to the checkpoint file")
@click.option("-r", "--run-name-prefix",      type=str, default="test_c2cbi_scz", help="Run name prefix for logging")
@click.option("-c", "--use-colab",            is_flag=True,                      help="Use Google Colab environment paths")
def main(*args, **kwargs):
    cfg = TrainConfig_C2CBi_SCZ(*args, **kwargs)
    return test_c2cbi_scz(cfg)


if __name__ == "__main__":
    main()
