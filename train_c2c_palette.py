from pathlib import Path

import torch

from configs import TrainConfig_C2C_Palette
from core.dataset.datasets import get_dataloaders
from core.models import TrainModel_C2C_Palette
from core.utils.repo_utils import get_repo_dir
from palette.models.palette_network import PaletteNetwork


def main():
    ROOT_IMAGE_DIR = get_repo_dir() / Path("data/datasets/paired_32x32")
    METADATA_PATH = ROOT_IMAGE_DIR / "metadata.jsonl"

    # Config
    cfg = TrainConfig_C2C_Palette(
        image_size=32,
        train_batch_size=16,
        eval_batch_size=8,
        eval_epoch_interval=1,
        checkpoint_epoch_interval=5,
    )

    print(f"Starting Char2Char training with Palette model:")
    print(f"    Image size: {cfg.image_size}")
    print(f"    Batch size: {cfg.train_batch_size}")
    print(f"    Epochs: {cfg.num_epochs}")
    print(f"    Learning rate: {cfg.learning_rate}")

    # Dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        cfg, 
        root_image_dir=ROOT_IMAGE_DIR, 
        metadata_path=METADATA_PATH
    )

    # Network
    net = PaletteNetwork(config=cfg)
    net.init_weights()

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable network parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate)

    # Model
    training_model = TrainModel_C2C_Palette(
        config=cfg,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        net=net,
        optimizer=optimizer,
        lr_scheduler=None,
    )

    # Training loop
    training_model.train()


if __name__ == "__main__":
    main()
