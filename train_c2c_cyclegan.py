import itertools

import rich_click as click
import torch

from configs import TrainConfig_C2C_CycleGAN
from core.models.c2c_cyclegan import CycleGANNetwork, TrainModel_C2C_CycleGAN


def train_c2c_cyclegan(cfg: TrainConfig_C2C_CycleGAN):
    print(f"Starting Char2Char training with CycleGAN model:")
    print(f"    Dataset: {cfg.root_image_dir.name}")
    print(f"    Image size: {cfg.image_size}")
    print(f"    Batch size: {cfg.train_batch_size}")
    print(f"    Epochs: {cfg.num_epochs}")
    print(f"    Learning rate: {cfg.learning_rate}")

    # Create the composite network
    net = CycleGANNetwork(cfg)

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable network parameters: {total_params:,}")

    # Create optimizers for generators and discriminators
    optimizer_G = torch.optim.Adam(
        itertools.chain(net.netG_A.parameters(), net.netG_B.parameters()), 
        lr=cfg.learning_rate, betas=(0.5, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        itertools.chain(net.netD_A.parameters(), net.netD_B.parameters()), 
        lr=cfg.learning_rate, betas=(0.5, 0.999)
    )

    training_model = TrainModel_C2C_CycleGAN(
        config=cfg,
        net=net,
        optimizer=optimizer_G,  # Generator optimizer goes to base class
        lr_scheduler=None,
        optimizer_D=optimizer_D,  # Discriminator optimizer goes separately
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
@click.option("-rs",  "--seed",                      type=int,     help="Seed for random number generators")
@click.option("-si",  "--log-step-interval",         type=int,     help="Log metrics every N steps")
@click.option("-ei",  "--eval-epoch-interval",       type=int,     help="Run validation every N epochs")
@click.option("-ci",  "--checkpoint-epoch-interval", type=int,     help="Save model checkpoints every N epochs")
@click.option("-c",   "--use-colab",                 is_flag=True, help="Use Google Colab environment paths")
@click.option("-p",   "--load-checkpoint-path",      type=str,     help="Path to load model checkpoint from")
@click.option("--lambda-a",                          type=float,   help="Weight for cycle loss (A -> B -> A)")
@click.option("--lambda-b",                          type=float,   help="Weight for cycle loss (B -> A -> B)")
@click.option("--lambda-identity",                   type=float,   help="Weight for identity mapping loss")
@click.option("--netg",                              type=str,     help="Generator architecture")
@click.option("--netd",                              type=str,     help="Discriminator architecture")
@click.option("--gan-mode",                          type=str,     help="GAN loss type")
def main(**kwargs):
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    cfg = TrainConfig_C2C_CycleGAN(**filtered_kwargs)
    return train_c2c_cyclegan(cfg)


if __name__ == "__main__":
    main()
