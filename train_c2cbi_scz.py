from pathlib import Path

import rich_click as click
import torch
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup

from configs import TrainConfig_C2CBi_SCZ
from core.dataset.datasets import get_dataloaders
from core.models import TrainModel_C2CBi_SCZ


def train_c2cbi_scz(**config_kwargs):
    # Config
    cfg = TrainConfig_C2CBi_SCZ(**config_kwargs)

    print(f"Starting Char2CharBi training with ShengChengZi model:")
    print(f"    Dataset: {cfg.root_image_dir.name}")
    print(f"    Image size: {cfg.image_size}")
    print(f"    Batch size: {cfg.train_batch_size}")
    print(f"    Epochs: {cfg.num_epochs}")
    print(f"    Learning rate: {cfg.learning_rate}")

    # Dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        cfg, 
        root_image_dir=cfg.root_image_dir, 
        metadata_path=cfg.image_metadata_path
    )

    # Network
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

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable network parameters: {total_params:,}")

    # Optimizer and schedulers
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=len(train_dataloader) * cfg.num_epochs,
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Model
    training_model = TrainModel_C2CBi_SCZ(
        config=cfg,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        net=net,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        noise_scheduler=noise_scheduler,
    )

    # Training loop
    training_model.train()

    return training_model


@click.command()
@click.option("--image-size", default=32, type=int, help="Image resolution")
@click.option("--validation-split", default=16, type=click.FLOAT, help="Validation split (0.0-1.0 or int for absolute count)")
@click.option("--test-split", default=16, type=click.FLOAT, help="Test split (0.0-1.0 or int for absolute count)")
@click.option("--num-epochs", default=100, type=int, help="Training epochs")
@click.option("--train-batch-size", default=16, type=int, help="Training batch size (number of images)")
@click.option("--eval-batch-size", default=16, type=int, help="Evaluation batch size (number of images)")
@click.option("--learning-rate", default=1e-4, type=float, help="Model learning rate")
@click.option("--seed", default=0, type=int, help="Seed for random number generators")
@click.option("--log-step-interval", default=1, type=int, help="Log metrics every N steps")
@click.option("--eval-epoch-interval", default=1, type=int, help="Run validation every N epochs")
@click.option("--checkpoint-epoch-interval", default=5, type=int, help="Save model checkpoints every N epochs")
@click.option("--use-colab", is_flag=True, help="Use Google Colab environment paths")
def main(*args, **kwargs):
    train_c2cbi_scz(*args, **kwargs)


if __name__ == "__main__":
    main()
