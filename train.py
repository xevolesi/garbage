import os
import argparse as ap

import yaml
import pandas as pd
from names_generator import generate_name

from source.config import Config


def read_config(path: str) -> Config:
    with open(path, "r") as yaml_file:
        yml = yaml.safe_load(yaml_file)
    return Config.model_validate(yml)


def main(args: ap.Namespace) -> None:
    config = read_config(args.config)
    if config.path.run_name is None:
        config.path.run_name = generate_name(seed=config.hyperparams.seed)
    config.path.artifacts_folder = os.path.join(
        config.path.artifacts_folder, config.path.run_name
    )
    os.makedirs(config.path.artifacts_folder, exist_ok=True)
    dataframe = pd.read_csv(
        os.path.join(config.path.base_dataset_folder, config.path.csv_name)
    )
    






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