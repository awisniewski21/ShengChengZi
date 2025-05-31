import importlib
from functools import partial
from types import FunctionType
from typing import Dict, Tuple

import numpy as np
from torch import Generator, randperm
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from palette.utils.device_utils import set_seed
from palette.utils.logger import InfoLogger


def define_dataloader(opt: Dict, logger: InfoLogger):
    """ Create train & validation dataloaders, or a test dataloader """
    # Set up dataloader arguments and worker seed
    dataloader_args = opt["datasets"][opt["phase"]]["dataloader"]["kwargs"]
    worker_init_fn = partial(set_seed, gl_seed=opt["seed"])

    # Create datasets
    phase_dataset, val_dataset = define_dataset(opt, logger)

    # Create data sampler for distributed training
    data_sampler = None
    if opt["distributed"]:
        data_sampler = DistributedSampler(phase_dataset, shuffle=dataloader_args.get("shuffle", False), num_replicas=opt["world_size"], rank=opt["global_rank"])
        dataloader_args.update({"shuffle": False}) # Sampler and shuffle are mutually exclusive

    # Create main dataloader
    dataloader = DataLoader(phase_dataset, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)

    # Create validation dataloader (only on GPU 0)
    if opt["global_rank"] == 0 and val_dataset is not None:
        val_args = opt["datasets"][opt["phase"]]["dataloader"].get("val_args", {})
        dataloader_args.update(val_args)
        val_dataloader = DataLoader(val_dataset, worker_init_fn=worker_init_fn, **dataloader_args)
    else:
        val_dataloader = None

    return dataloader, val_dataloader


def define_dataset(opt: Dict, logger: InfoLogger):
    """
    Load a dataset from a given file with an optiona train/validation split
    """
    dataset_opt = opt["datasets"][opt["phase"]]["which_dataset"]
    phase_dataset = init_obj(dataset_opt, logger, "palette.dataset")
    val_dataset = None

    data_len = len(phase_dataset)
    valid_len = 0

    # Debug split
    if "debug" in opt["name"]:
        debug_split = opt["debug"].get("debug_split", 1.0)
        data_len = debug_split if isinstance(debug_split, int) else int(data_len * debug_split)

    dataloader_opt = opt["datasets"][opt["phase"]]["dataloader"]
    valid_split = dataloader_opt.get("validation_split", 0)

    # Validation split
    if valid_split > 0.0 or "debug" in opt["name"]:
        if isinstance(valid_split, int):
            assert valid_split < data_len, "Validation set size is configured to be larger than entire dataset."
            valid_len = valid_split
        else:
            valid_len = int(data_len * valid_split)
        data_len -= valid_len
        phase_dataset, val_dataset = subset_split(dataset=phase_dataset, lengths=[data_len, valid_len], generator=Generator().manual_seed(opt["seed"]))

    logger.info(f"Dataset for {opt['phase']} has {data_len} samples.")
    if opt["phase"] == "train":
        logger.info(f"Dataset for val has {valid_len} samples.")

    return phase_dataset, val_dataset


def define_model(opt: Dict, logger: InfoLogger, **model_kwargs):
    """ Create a model instance based on the configuration options """
    model_opt = opt["model"]["which_model"]
    model_opt["kwargs"].update(model_kwargs)
    return init_obj(model_opt, logger, "palette.models", opt=opt, logger=logger)


def define_network(network_opt: Dict, opt: Dict, logger: InfoLogger):
    """ Create a network instance based on the configuration options """
    net = init_obj(network_opt, logger, "palette.models")

    if opt["phase"] == "train":
        logger.info(f"Network weights for '{net.__class__.__name__}' initialized using '{network_opt['kwargs'].get('init_type', 'default')}'")
        net.init_weights()

    return net


def define_loss(opt: Dict, logger: InfoLogger):
    return init_obj(opt, logger, "palette.models")


def define_metric(opt: Dict, logger: InfoLogger):
    return init_obj(opt, logger, "palette.models")


###
### Helper Functions
###

def init_obj(this_opt: str | Dict, this_logger: InfoLogger, module_name: str, **module_kwargs):
    """
    Loads a class or function by name from a module and initializes it with the given arguments in "opt".
    """
    if not this_opt:
        return None

    if isinstance(this_opt, str):
        this_opt = {"name": this_opt}

    try:
        module = importlib.import_module(module_name)
        attr = getattr(module, this_opt["name"])
        kwargs = this_opt.get("kwargs", {})
        kwargs.update(module_kwargs)

        if isinstance(attr, type):
            obj = attr(**kwargs)
            obj.__name__ = obj.__class__.__name__
        elif isinstance(attr, FunctionType):
            obj = partial(attr, **kwargs)
            obj.__name__ = attr.__name__

        this_logger.info(f"Loaded '{this_opt['name']}' from module '{module_name}'")
    except Exception:
        raise NotImplementedError(f"'{this_opt['name']}' not found in module '{module_name}'")

    return obj


def subset_split(dataset: Dataset, lengths: Tuple[int, int], generator: Generator):
    """
    Split a dataset into non-overlapping new datasets of given lengths.
    """
    indices = randperm(sum(lengths), generator=generator).tolist()
    subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths):
        if length == 0:
            subsets.append(None)
        else:
            subsets.append(Subset(dataset, indices[offset - length: offset]))
    return subsets

