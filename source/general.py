import os
import random

import numpy as np
import torch
import yaml

from .config import Config


def read_config(path: str) -> Config:
    with open(path, "r") as yaml_file:
        yml = yaml.safe_load(yaml_file)
    return Config.model_validate(yml)


def get_cpu_state_dict(state_dict):
    return {name: tensor.detach().cpu() for name, tensor in state_dict.items()}


def load_optimizer_state_dict(optimizer, state_dict, device):
    optimizer.load_state_dict(state_dict)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def seed_everything(config: Config, local_rank: int = 0) -> None:
    """
    Fix all avaliable seeds to ensure reproducibility.
    Local rank is needed for distributed data parallel training.
    It is used to make seeds different for different processes.
    Each process will have `seed = config_seed + local_rank`.
    """
    random.SystemRandom().seed(config.training.seed + local_rank)
    np.random.seed(config.training.seed + local_rank)
    torch.manual_seed(config.training.seed + local_rank)
    os.environ["PYTHONHASHSEED"] = str(config.training.seed + local_rank)
    torch.cuda.manual_seed(config.training.seed + local_rank)
    torch.cuda.manual_seed_all(config.training.seed + local_rank)
