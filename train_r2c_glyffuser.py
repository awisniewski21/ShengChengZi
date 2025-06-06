from pathlib import Path

import rich_click as click
import torch
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup

from configs import TrainConfig_R2C_Glyff
from core.dataset.datasets import get_dataloaders
from core.models import TrainModel_R2C_Glyffuser


def train_r2c_glyffuser(cfg: TrainConfig_R2C_Glyff):
    print(f"Starting Rand2Char training with Glyffuser model:")
    print(f"    Dataset: {cfg.root_image_dir.name}")
    print(f"    Image size: {cfg.image_size}")
    print(f"    Batch size: {cfg.train_batch_size}")
    print(f"    Epochs: {cfg.num_epochs}")
    print(f"    Learning rate: {cfg.learning_rate}")

    train_dataloader, _, _ = get_dataloaders(
        cfg, 
        root_image_dir=cfg.root_image_dir, 
        metadata_path=cfg.image_metadata_path,
    )

    net = UNet2DModel(
        sample_size=cfg.image_size,
        in_channels=1,
        out_channels=1,
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels=(32, 64, 128, 128),
        layers_per_block=2,
    )

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable network parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=len(train_dataloader) * cfg.num_epochs,
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    inference_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000)

    training_model = TrainModel_R2C_Glyffuser(
        config=cfg,
        net=net,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        noise_scheduler=noise_scheduler,
        inference_scheduler=inference_scheduler,
    )
    if cfg.load_checkpoint_path is not None:
        training_model.load_checkpoint("train")

    training_model.train()

    return training_model


@click.command()
@click.option("-s",   "--image-size",                type=int,     help="Image resolution")
@click.option("-vs",  "--validation-split",          type=float,   help="Validation split (0.0-1.0 or int for absolute count)")
@click.option("-ts",  "--test-split",                type=float,   help="Test split (0.0-1.0 or int for absolute count)")
@click.option("-e",   "--num-epochs",                type=int,     help="Training epochs")
@click.option("-tbs", "--train-batch-size",          type=int,     help="Training batch size (number of images)")
@click.option("-ebs", "--eval-batch-size",           type=int,     help="Evaluation batch size (number of images)")
@click.option("-lr",  "--learning-rate",             type=float,   help="Model learning rate")
@click.option("-s",   "--seed",                      type=int,     help="Seed for random number generators")
@click.option("-si",  "--log-step-interval",         type=int,     help="Log metrics every N steps")
@click.option("-ei",  "--eval-epoch-interval",       type=int,     help="Run validation every N epochs")
@click.option("-ci",  "--checkpoint-epoch-interval", type=int,     help="Save model checkpoints every N epochs")
@click.option("-c",   "--use-colab",                 is_flag=True, help="Use Google Colab environment paths")
def main(**kwargs):
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    cfg = TrainConfig_R2C_Glyff(**filtered_kwargs)
    return train_r2c_glyffuser(cfg)


if __name__ == "__main__":
    main()
