from typing import Sequence

import rich_click as click

from configs import TrainConfig_C2C_Pix2Pix
from core.models.c2c_pix2pix import TrainModel_C2C_Pix2Pix, Pix2PixNetwork
from core.utils.image_utils import chars_to_image_tensor


def inference_c2c_pix2pix(
    cfg: TrainConfig_C2C_Pix2Pix,
    input_chars: Sequence[str],
    font_name: str = "NotoSansSC-Regular",
    font_size: int | None = None,
):
    assert cfg.load_checkpoint_path is not None, "Checkpoint path must be provided for inference"

    src_imgs = chars_to_image_tensor(input_chars, cfg.image_size, font_name, font_size)

    # Create the composite network
    net = Pix2PixNetwork(cfg)

    model = TrainModel_C2C_Pix2Pix(
        config=cfg,
        net=net,
        optimizer=None,
        lr_scheduler=None,
        optimizer_D=None,
    )
    model.load_checkpoint("inference")

    model.inference_step(src_imgs)


@click.command()
@click.argument(    "input_chars",            type=str, required=True, nargs=-1)
@click.option("-p", "--load-checkpoint-path", type=str, required=True,                   help="Path to the checkpoint file")
@click.option("-r", "--run-name-prefix",      type=str, default="inference_c2c_pix2pix", help="Run name prefix for logging")
@click.option("-c", "--use-colab",            is_flag=True,                              help="Use Google Colab environment paths")
@click.option("-f", "--font-name",            type=str, default="NotoSansSC-Regular",   help="Font name for rendering input characters")
@click.option("-fs", "--font-size",           type=int,                                  help="Font size for rendering input characters")
def main(input_chars, **kwargs):
    cfg = TrainConfig_C2C_Pix2Pix(**kwargs)
    return inference_c2c_pix2pix(cfg, input_chars, kwargs.get('font_name', 'NotoSansSC-Regular'), kwargs.get('font_size'))


if __name__ == "__main__":
    main()
