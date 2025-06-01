#!/usr/bin/env python3
"""
Palette model runner script.
"""

import os
import warnings

import click
import torch
import torch.multiprocessing as mp

from palette.utils.device_utils import set_seed
from palette.utils.load_modules import define_dataloader, define_model  # NOQA
from palette.utils.logger import InfoLogger, MetricsLogger
from palette.utils.parser import parse_dataclass_args
from configs.c2c_palette import TrainConfig_C2C_Palette


def main_worker(gpu: int, ngpus_per_node: int, config: TrainConfig_C2C_Palette):
    """
    Main function to run on each thread / GPU
    """
    if config.local_rank is None:
        config.local_rank = config.global_rank = gpu

    if config.distributed:
        torch.cuda.set_device(int(config.local_rank))
        print(f"Using GPU {int(config.local_rank)} for training")
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=config.init_method,
            world_size=config.world_size,
            rank=config.global_rank,
            group_name="mtorch",
        )

    torch.backends.cudnn.enabled = True
    warnings.warn("Using cuDNN for acceleration (torch.backends.cudnn.enabled=True)")
    set_seed(config.seed)

    logger = InfoLogger(config)
    writer = MetricsLogger(config, logger)
    logger.info(f"Created log file at '{writer.log_dir}'")

    phase_loader, val_loader = define_dataloader(config, logger)

    model = define_model(
        config,
        logger,
        phase_loader=phase_loader,
        val_loader=val_loader,
        writer=writer
    )

    logger.info(f"Executing phase '{config.phase}' for model")
    try:
        if config.phase == "train":
            model.train()
        else:
            model.test()
    finally:
        writer.close()


@click.command()
@click.option("-dr",  "--data_root", type=str,     default=None,                             help="Root directory for dataset")
@click.option("-p",   "--phase",     type=str,     default="train",                          help="Model phase ('train' or 'test')")
@click.option("-b",   "--batch",     type=int,     default=None,                             help="Batch size on every GPU")
@click.option("-gpu", "--gpu_ids",   type=str,     default=None,                             help="GPU IDs to use (e.g., '0,1,2')")
@click.option("-d",   "--debug",     is_flag=True,                                           help="Enable debug mode")
@click.option("-P",   "--port",      type=str,     default="21012",                          help="Port for distributed training")
def main(data_root: str, phase: str, batch: int, gpu_ids: str, debug: bool, port: str):
    # Create configuration from dataclass instead of JSON
    config = parse_dataclass_args(
        root_image_dir=data_root,
        phase=phase, 
        batch=batch, 
        gpu_ids=gpu_ids, 
        debug=debug
    )

    if len(config.gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(config.gpu_ids)
        print(f"Using GPUs: {config.gpu_ids}")
    else:
        print("No GPUs specified - using CPU for training")

    if config.distributed:
        ngpus_per_node = len(config.gpu_ids)
        config.world_size = ngpus_per_node
        config.init_method = f"tcp://127.0.0.1:{port}"
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        config.world_size = 1
        main_worker(0, 1, config)


if __name__ == "__main__":
    main()
