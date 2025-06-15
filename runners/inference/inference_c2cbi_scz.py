from typing import Sequence

import rich_click as click
import torch
from diffusers import DDPMScheduler, UNet2DModel

from core.configs import TrainConfig_C2CBi_SCZ
from core.models.c2cbi_scz import TrainModel_C2CBi_SCZ
from core.utils.image_utils import chars_to_image_tensor


def inference_c2cbi_scz(
    cfg: TrainConfig_C2CBi_SCZ,
    input_chars: Sequence[str],
    direction: int = 0,
    font_name: str = "NotoSansSC-Regular",
    font_size: int | None = None,
):
    assert cfg.load_checkpoint_path is not None, "Checkpoint path must be provided for inference"

    src_imgs = chars_to_image_tensor(input_chars, cfg, font_name, font_size)
    labels = torch.tensor([direction] * len(input_chars))

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
    model.load_checkpoint("inference")

    input_data = (src_imgs, labels)
    model.inference_step(input_data)


@click.command()
@click.argument(    "input_chars",            type=str, required=True, nargs=-1)
@click.option("-p", "--load-checkpoint-path", type=str, required=True,                 help="Path to the checkpoint file")
@click.option("-r", "--run-name-prefix",      type=str, default="inference_c2cbi_scz", help="Run name prefix for logging")
@click.option("-c", "--use-colab",            is_flag=True,                            help="Use Google Colab environment paths")
@click.option("-d", "--direction",            type=int, default=0,                     help="Generation direction (0 for src->trg or 1 for trg->src)")
@click.option("-f", "--font-name",            type=str, default="NotoSansSC-Regular",  help="Font name for character rendering")
@click.option("-s", "--font-size",            type=int, default=None,                  help="Font size for character rendering (None to auto-scale)")
def main(*args, input_chars: Sequence[str], direction: int, font_name: str, font_size: int | None, **kwargs):
    cfg = TrainConfig_C2CBi_SCZ(*args, **kwargs)
    return inference_c2cbi_scz(cfg, input_chars, direction=direction, font_name=font_name, font_size=font_size)


if __name__ == "__main__":
    main()
