from typing import Sequence

import rich_click as click

from core.configs import TrainConfig_C2C_Palette
from core.models.c2c_palette import TrainModel_C2C_Palette
from core.utils.image_utils import chars_to_image_tensor
from palette.palette_network import PaletteNetwork


def inference_c2c_palette(
    cfg: TrainConfig_C2C_Palette,
    input_chars: Sequence[str],
    font_name: str = "NotoSansSC-Regular",
    font_size: int | None = None,
):
    assert cfg.load_checkpoint_path is not None, "Checkpoint path must be provided for evaluation"

    src_imgs = chars_to_image_tensor(input_chars, cfg.image_size, font_name, font_size)

    net = PaletteNetwork(config=cfg)

    model = TrainModel_C2C_Palette(
        config=cfg,
        net=net,
        optimizer=None,
        lr_scheduler=None,
    )
    model.load_checkpoint("inference")

    model.inference(src_imgs)


@click.command()
@click.argument(    "input_chars",            type=str, required=True, nargs=-1)
@click.option("-p", "--load-checkpoint-path", type=str, required=True,                   help="Path to the checkpoint file")
@click.option("-r", "--run-name-prefix",      type=str, default="inference_c2c_palette", help="Run name prefix for logging")
@click.option("-c", "--use-colab",            is_flag=True,                              help="Use Google Colab environment paths")
@click.option("-f", "--font-name",            type=str, default="NotoSansSC-Regular",    help="Font name for character rendering")
@click.option("-s", "--font-size",            type=int, default=None,                    help="Font size for character rendering (None to auto-scale)")
def main(*args, input_chars: Sequence[str], font_name: str, font_size: int | None, **kwargs):
    cfg = TrainConfig_C2C_Palette(*args, **kwargs)
    return inference_c2c_palette(cfg, input_chars, font_name=font_name, font_size=font_size)


if __name__ == "__main__":
    main()
