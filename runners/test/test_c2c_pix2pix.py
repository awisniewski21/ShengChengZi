import rich_click as click

from core.configs import TrainConfig_C2C_Pix2Pix
from core.models.c2c_pix2pix import TrainModel_C2C_Pix2Pix, Pix2PixNetwork


def test_c2c_pix2pix(cfg: TrainConfig_C2C_Pix2Pix):
    assert cfg.load_checkpoint_path is not None, "Checkpoint path must be provided for evaluation"

    # Create the composite network
    net = Pix2PixNetwork(cfg)

    model = TrainModel_C2C_Pix2Pix(
        config=cfg,
        net=net,
        optimizer=None,
        lr_scheduler=None,
        optimizer_D=None,
    )
    model.load_checkpoint("test")

    test_metrics = model.test()

    return test_metrics


@click.command()
@click.option("-p", "--load-checkpoint-path", type=str, required=True,              help="Path to the checkpoint file")
@click.option("-r", "--run-name-prefix",      type=str, default="test_c2c_pix2pix", help="Run name prefix for logging")
@click.option("-c", "--use-colab",            is_flag=True,                         help="Use Google Colab environment paths")
def main(*args, **kwargs):
    cfg = TrainConfig_C2C_Pix2Pix(*args, **kwargs)
    return test_c2c_pix2pix(cfg)


if __name__ == "__main__":
    main()
