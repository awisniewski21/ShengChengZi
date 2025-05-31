import os
import warnings
from typing import Dict

import click
import torch
import torch.multiprocessing as mp
from core.load_modules import define_dataloader, define_loss, define_metric, define_model, define_network  # NOQA
from core.logger import InfoLogger, MetricsLogger
from core.parser import parse_cli_args
from core.utils import set_seed


def main_worker(gpu: int, ngpus_per_node: int, opt: Dict):
    """
    Main function to run on each thread / GPU
    """
    if "local_rank" not in opt:
        opt["local_rank"] = opt["global_rank"] = gpu

    if opt["distributed"]:
        torch.cuda.set_device(int(opt["local_rank"]))
        print(f"Using GPU {int(opt['local_rank'])} for training")
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=opt["init_method"],
            world_size=opt["world_size"],
            rank=opt["global_rank"],
            group_name="mtorch",
        )

    torch.backends.cudnn.enabled = True
    warnings.warn("Using cuDNN for acceleration (torch.backends.cudnn.enabled=True)")
    set_seed(opt["seed"])

    logger = InfoLogger(opt)
    writer = MetricsLogger(opt, logger)
    logger.info(f"Created log file at '{opt['path']['experiments_root']}'")

    phase_loader, val_loader = define_dataloader(opt, logger)

    networks = [define_network(net_opt, opt, logger) for net_opt in opt["model"]["which_networks"]]

    metrics = [define_metric(metric_opt, logger) for metric_opt in opt["model"]["which_metrics"]]
    losses = [define_loss(loss_opt, logger) for loss_opt in opt["model"]["which_losses"]]

    model = define_model(
        opt,
        logger,
        networks=networks,
        phase_loader=phase_loader,
        val_loader=val_loader,
        losses=losses,
        metrics=metrics,
        writer=writer,
    )

    logger.info(f"Executing phase '{opt['phase']}' for model")
    try:
        if opt["phase"] == "train":
            model.train()
        else:
            model.test()
    finally:
        writer.close()


@click.command()
@click.option("-c",   "--config",  type=str,     default="config/char2char.json", help="JSON file for configuration")
@click.option("-p",   "--phase",   type=str,     default="train",                 help="Model phase ('train' or 'test')")
@click.option("-b",   "--batch",   type=int,     default=None,                    help="Batch size on every GPU")
@click.option("-gpu", "--gpu_ids", type=str,     default=None,                    help="GPU IDs to use (e.g., '0,1,2')")
@click.option("-d",   "--debug",   is_flag=True,                                  help="Enable debug mode")
@click.option("-P",   "--port",    type=str,     default="21012",                 help="Port for distributed training")
def main(config: str, phase: str, batch: int, gpu_ids: str, debug: bool, port: str):
    opt = parse_cli_args(config, phase, batch, gpu_ids, debug)

    if len(opt["gpu_ids"]) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(opt["gpu_ids"])
        print(f"Using GPUs: {opt['gpu_ids']}")
    else:
        print("No GPUs specified - using CPU for training")

    if opt["distributed"]:
        ngpus_per_node = len(opt["gpu_ids"])
        opt["world_size"] = ngpus_per_node
        opt["init_method"] = f"tcp://127.0.0.1:{port}"
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt["world_size"] = 1
        main_worker(0, 1, opt)


if __name__ == "__main__":
    main()