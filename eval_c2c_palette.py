import rich_click as click

from configs import TrainConfig_C2C_Palette
from core.models.c2c_palette import TrainModel_C2C_Palette
from palette.models.palette_network import PaletteNetwork


def eval_c2c_palette(cfg: TrainConfig_C2C_Palette):
    assert cfg.load_checkpoint_path is not None, "Checkpoint path must be provided for evaluation"

    net = PaletteNetwork(config=cfg)

    model = TrainModel_C2C_Palette(
        config=cfg,
        net=net,
        optimizer=None,
        lr_scheduler=None,
    )
    model.load_checkpoint("test")

    test_metrics = model.test()

    return test_metrics


@click.command()
@click.option("-p", "--load-checkpoint-path", type=str, required=True,              help="Path to the checkpoint file")
@click.option("-r", "--run-name-prefix",      type=str, default="eval_c2c_palette", help="Run name prefix for logging")
@click.option("-c", "--use-colab",            is_flag=True,                         help="Use Google Colab environment paths")
def main(*args, **kwargs):
    cfg = TrainConfig_C2C_Palette(*args, **kwargs)
    return eval_c2c_palette(cfg)


if __name__ == "__main__":
    main()
