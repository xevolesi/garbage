from pydantic import BaseModel


class PathConfig(BaseModel):
    run_name: str | None = None
    base_dataset_folder: str
    artifacts_folder: str
    csv_name: str


class DatasetConfig(BaseModel):
    train_folds: list[int]
    val_folds: list[int]


class HyperparamsConfig(BaseModel):
    seed: int


class Config(BaseModel):
    path: PathConfig
    dataset: DatasetConfig
    hyperparams: HyperparamsConfig
