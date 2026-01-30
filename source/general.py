import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import Config


def ensure_correct_path(path_str: str) -> Path:
    """
    Ensures a path is absolute, resolved, and uses the correct format
    for the current operating system.

    Args:
        path_str: The input path, which can be relative or contain
                  shell variables/symlinks.

    Returns:
        A pathlib.Path object representing the absolute, resolved path.
    """
    # Expand shell variables (like $HOME) and user home directory (~user)
    expanded_path = os.path.expandvars(os.path.expanduser(path_str))

    # Create a Path object
    p = Path(expanded_path)

    # Resolve the path:
    # 1. Makes it absolute if it's relative to the current working directory.
    # 2. Collapses redundant separators (A//B becomes A/B).
    # 3. Processes ".." and "." components (A/foo/../B becomes A/B).
    # 4. Follows symbolic links to the actual file/directory.
    try:
        resolved_path = p.resolve()
        return resolved_path
    except FileNotFoundError:
        # If the file or an intermediate directory doesn't exist, resolve() fails.
        # In this case, we return an absolute path without existence check.
        # This is useful if you want a correct path to a *potential* file.
        return p.absolute()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return p.absolute()  # Fallback


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
