import rich_click as click

from core.configs import TrainConfig_C2CBi_CycleGAN
from core.models.c2cbi_cyclegan import TrainModel_C2CBi_CycleGAN, CycleGANNetwork


def test_c2cbi_cyclegan(cfg: TrainConfig_C2CBi_CycleGAN):
    assert cfg.load_checkpoint_path is not None, "Checkpoint path must be provided for evaluation"

    # Create the composite network
    net = CycleGANNetwork(cfg)

    model = TrainModel_C2CBi_CycleGAN(
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
@click.option("-p", "--load-checkpoint-path", type=str, required=True,                 help="Path to the checkpoint file")
@click.option("-r", "--run-name-prefix",      type=str, default="test_c2cbi_cyclegan", help="Run name prefix for logging")
@click.option("-c", "--use-colab",            is_flag=True,                            help="Use Google Colab environment paths")
def main(*args, **kwargs):
    cfg = TrainConfig_C2CBi_CycleGAN(*args, **kwargs)
    return test_c2cbi_cyclegan(cfg)


if __name__ == "__main__":
    main()
