import argparse
import json
import os
import shutil
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict


def parse_cli_args(config: str, phase: str, batch: int, gpu_ids: str, debug: bool) -> Dict:
    """
    Parse the configuration file and command-line arguments, set up directories, and backup code.
    """
    # Read and clean JSON config (remove comments)
    json_str = ""
    with open(config, "r") as f:
        for line in f:
            line = line.split("//")[0] + "\n"
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    # Replace config context using args
    opt["phase"] = phase
    if gpu_ids is not None:
        opt["gpu_ids"] = [int(id) for id in gpu_ids.split(",")]
    if batch is not None:
        opt["datasets"][opt["phase"]]["dataloader"]["kwargs"]["batch_size"] = batch

    # Set CUDA environment
    opt["distributed"] = opt["gpu_ids"] is not None and len(opt["gpu_ids"]) > 1

    # Update experiment name
    prefix = "debug" if debug else ("finetune" if opt.get("finetune_norm") else opt['phase'])
    opt["name"] = f"{prefix}_{opt['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Update training options if debugging
    if "debug" in opt["name"]:
        opt["train"].update(opt["debug"])

    return dict_to_nonedict(opt)


def dict_to_nonedict(opt: Dict):
    """
    Recursively convert dicts to defaultdicts that return None for missing keys.
    """
    if isinstance(opt, dict):
        new_opt = defaultdict(lambda: None)
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return new_opt
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def write_json(out_dict: Dict, out_file: str):
    """Write a dictionary to a JSON file."""
    with Path(out_file).open("wt") as handle:
        json.dump(out_dict, handle, indent=4, sort_keys=False)