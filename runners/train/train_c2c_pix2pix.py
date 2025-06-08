import rich_click as click
import torch

from core.configs import TrainConfig_C2C_Pix2Pix
from core.models.c2c_pix2pix import Pix2PixNetwork, TrainModel_C2C_Pix2Pix


def train_c2c_pix2pix(cfg: TrainConfig_C2C_Pix2Pix):
    print(f"Starting Char2Char training with Pix2Pix model:")
    print(f"    Dataset: {cfg.root_image_dir.name}")
    print(f"    Image size: {cfg.image_size}")
    print(f"    Batch size: {cfg.train_batch_size}")
    print(f"    Epochs: {cfg.num_epochs}")
    print(f"    Learning rate: {cfg.learning_rate}")

    net = Pix2PixNetwork(cfg)

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable network parameters: {total_params:,}")

    optimizer_G = torch.optim.Adam(net.netG.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(net.netD.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))

    training_model = TrainModel_C2C_Pix2Pix(
        config=cfg,
        net=net,
        optimizer=optimizer_G,
        lr_scheduler=None,
        optimizer_D=optimizer_D,
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
@click.option("-p",   "--load-checkpoint-path",      type=float,   help="Optional path to load model checkpoint from")
@click.option("-rs",  "--seed",                      type=int,     help="Seed for random number generators")
@click.option("-c",   "--use-colab",                 is_flag=True, help="Use Google Colab environment paths")
@click.option("-si",  "--log-step-interval",         type=int,     help="Log metrics every N steps")
@click.option("-ei",  "--eval-epoch-interval",       type=int,     help="Run validation every N epochs")
@click.option("-ci",  "--checkpoint-epoch-interval", type=int,     help="Save model checkpoints every N epochs")
def main(**kwargs):
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    cfg = TrainConfig_C2C_Pix2Pix(**filtered_kwargs)
    return train_c2c_pix2pix(cfg)


if __name__ == "__main__":
    main()
