from pathlib import Path, PosixPath

import rich_click as click
import torch

from configs.c2c_palette import TrainConfig_C2C_Palette
from core.dataset.datasets import get_dataloaders
from core.models.c2c_palette import TrainModel_C2C_Palette
from core.utils.repo_utils import get_repo_dir
from palette.models.palette_network import PaletteNetwork


@click.command()
@click.option("-c", "--checkpoint-path", required=True, type=str, help="Path to the checkpoint file")
def eval_c2c_palette(checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.serialization.add_safe_globals([TrainConfig_C2C_Palette, PosixPath])
    chkpt_data = torch.load(checkpoint_path, map_location=device)
    config: TrainConfig_C2C_Palette = chkpt_data["config"]
    config.repo_dir = Path(get_repo_dir())
    config.use_colab = False

    print(f"Loading model from {checkpoint_path}")
    print(f"    Config seed: {config.seed}")
    print(f"    Epoch: {chkpt_data['current_epoch']}")

    # Create test dataloader using the saved config
    _, _, test_dataloader = get_dataloaders(
        config,
        root_image_dir=config.root_image_dir,
        metadata_path=config.image_metadata_path,
    )

    assert test_dataloader is not None, "No test set found for the loaded config!"

    # Create model instance and load checkpoint
    net = PaletteNetwork(config=config)

    model = TrainModel_C2C_Palette(
        config=config,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=test_dataloader,
        net=net,
        optimizer=None,
        lr_scheduler=None,
    )

    model.load_checkpoint(checkpoint_path, "test")

    # Run inference on test set
    test_metrics = model.test()

    return test_metrics


if __name__ == "__main__":
    eval_c2c_palette()
