import os
import argparse as ap

import yaml
import torch
import pandas as pd
from names_generator import generate_name

from source.config import Config
from source.dataset import build_dataloaders

from source.general import seed_everything
from source.models import YuNet


def read_config(path: str) -> Config:
    with open(path, "r") as yaml_file:
        yml = yaml.safe_load(yaml_file)
    return Config.model_validate(yml)


def train(config: Config, dataframe: pd.DataFrame):
    device = torch.device(config.training.device)
    dataloaders = build_dataloaders(config, dataframe)
    model = YuNet(**config.model.model_dump()).to(device)



def main(args: ap.Namespace) -> None:
    config = read_config(args.config)
    if config.path.run_name is None:
        config.path.run_name = generate_name(seed=config.training.seed)
    config.path.artifacts_folder = os.path.join(
        config.path.artifacts_folder, config.path.run_name
    )
    os.makedirs(config.path.artifacts_folder, exist_ok=True)
    dataframe = pd.read_csv(config.path.csv)

    seed_everything(config)
    train(config, dataframe)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    args = parser.parse_args()
    main(args)