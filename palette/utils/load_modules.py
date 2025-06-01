import numpy as np
from functools import partial
from torch import Generator, randperm
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from palette.utils.device_utils import set_seed
from palette.utils.logger import InfoLogger
from configs.char2char.palette import TrainConfig_C2C_Palette
from core.dataset.datasets import get_dataloader


def define_dataloader(config: TrainConfig_C2C_Palette, logger: InfoLogger):
    """ Create train & validation dataloaders, or a test dataloader """
    # Use core dataset functionality
    dataloader = get_dataloader(
        config,
        config.root_image_dir,
        batch_size=config.train_batch_size if config.phase == "train" else config.eval_batch_size,
        shuffle=(config.phase == "train")
    )
    
    # Create validation dataloader if needed (only for training phase and GPU 0)
    val_dataloader = None
    if config.phase == "train" and config.global_rank == 0 and config.validation_split > 0:
        # Get dataset for validation split
        full_dataset = dataloader.dataset
        data_len = len(full_dataset)
        
        valid_split = config.validation_split
        if isinstance(valid_split, int):
            assert valid_split < data_len, "Validation set size is configured to be larger than entire dataset."
            valid_len = valid_split
        else:
            valid_len = int(data_len * valid_split)
        
        data_len -= valid_len
        
        # Split dataset
        indices = randperm(len(full_dataset), generator=Generator().manual_seed(config.seed)).tolist()
        train_dataset = Subset(full_dataset, indices[:data_len])
        val_dataset = Subset(full_dataset, indices[data_len:data_len + valid_len])
        
        # Create new dataloaders
        worker_init_fn = partial(set_seed, gl_seed=config.seed)
        
        # Update training dataloader with train subset
        data_sampler = None
        if config.distributed:
            data_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=config.world_size, rank=config.global_rank)
        
        dataloader = DataLoader(
            train_dataset, 
            batch_size=config.train_batch_size,
            shuffle=(data_sampler is None),
            sampler=data_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            collate_fn=dataloader.collate_fn
        )
        
        # Create validation dataloader
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            collate_fn=dataloader.collate_fn
        )
        
        logger.info(f"Dataset for train has {data_len} samples.")
        logger.info(f"Dataset for val has {valid_len} samples.")
    else:
        # For distributed training without validation split
        if config.distributed and config.phase == "train":
            dataset = dataloader.dataset
            data_sampler = DistributedSampler(dataset, shuffle=True, num_replicas=config.world_size, rank=config.global_rank)
            worker_init_fn = partial(set_seed, gl_seed=config.seed)
            
            dataloader = DataLoader(
                dataset,
                batch_size=config.train_batch_size,
                sampler=data_sampler,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
                worker_init_fn=worker_init_fn,
                collate_fn=dataloader.collate_fn
            )
        
        logger.info(f"Dataset for {config.phase} has {len(dataloader.dataset)} samples.")

    return dataloader, val_dataloader


def define_model(config: TrainConfig_C2C_Palette, logger: InfoLogger, **model_kwargs):
    """ Create a model instance based on the configuration options """
    from palette.models import PaletteModel
    
    # Create network directly from config
    networks = [define_network(config, logger)]
    
    # Create losses directly from config
    losses = [define_loss(config, logger)]
    
    # Create metrics directly from config  
    metrics = [define_metric(config, logger)]
    
    # Create optimizer config
    optimizer_config = [{
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay
    }]
    
    # Create EMA scheduler if enabled
    ema_scheduler = None
    if config.ema_enabled:
        ema_scheduler = {
            "ema_start": config.ema_start,
            "ema_iter": config.ema_iter,
            "ema_decay": config.ema_decay
        }
    
    # Create model
    model = PaletteModel(
        networks=networks,
        losses=losses,
        sample_num=config.sample_num,
        optimizers=optimizer_config,
        ema_scheduler=ema_scheduler,
        config=config,
        logger=logger,
        metrics=metrics,
        **model_kwargs
    )
    
    return model


def define_network(config: TrainConfig_C2C_Palette, logger: InfoLogger):
    """ Create a network instance based on the configuration options """
    from palette.models import PaletteNetwork
    
    # Create UNet kwargs from config
    unet_kwargs = {
        "in_channel": config.in_channel,
        "out_channel": config.out_channel, 
        "inner_channel": config.inner_channel,
        "channel_mults": config.channel_mults,
        "attn_res": config.attn_res,
        "num_head_channels": config.num_head_channels,
        "res_blocks": config.res_blocks,
        "dropout": config.dropout,
        "image_size": config.image_size
    }
    
    # Create beta schedule kwargs from config
    beta_schedule_kwargs = {
        "train": {
            "schedule": config.schedule,
            "n_timestep": config.n_timestep_train,
            "linear_start": config.linear_start_train,
            "linear_end": config.linear_end_train
        },
        "test": {
            "schedule": config.schedule,
            "n_timestep": config.n_timestep_test,
            "linear_start": config.linear_start_test,
            "linear_end": config.linear_end_test
        }
    }
    
    # Create network
    net = PaletteNetwork(
        unet_kwargs=unet_kwargs,
        beta_schedule_kwargs=beta_schedule_kwargs,
        module_name=config.module_name,
        init_type=config.init_type
    )

    if config.phase == "train":
        logger.info(f"Network weights for '{net.__class__.__name__}' initialized using '{config.init_type}'")
        net.init_weights()

    return net


def define_loss(config: TrainConfig_C2C_Palette, logger: InfoLogger):
    """ Create a loss function based on the configuration """
    from palette.models import mse_loss, FocalLoss
    
    if config.loss_function == "mse_loss":
        loss_fn = mse_loss
    elif config.loss_function == "focal_loss":
        loss_fn = FocalLoss()
    else:
        raise NotImplementedError(f"Loss function '{config.loss_function}' not implemented")
    
    logger.info(f"Using loss function: {config.loss_function}")
    return loss_fn


def define_metric(config: TrainConfig_C2C_Palette, logger: InfoLogger):
    """ Create metric functions based on the configuration """
    from palette.models import mae, inception_score
    
    metrics = []
    for metric_name in config.metrics:
        if metric_name == "mae":
            metrics.append(mae)
        elif metric_name == "inception_score":
            metrics.append(inception_score)
        else:
            raise NotImplementedError(f"Metric '{metric_name}' not implemented")
    
    logger.info(f"Using metrics: {config.metrics}")
    return metrics[0] if len(metrics) == 1 else metrics

