import typing as ty

from pydantic import BaseModel


class PathConfig(BaseModel):
    run_name: str | None = None
    base_dataset_folder: str
    artifacts_folder: str
    csv: str


class DatasetConfig(BaseModel):
    image_path_col: str
    boxes_col: str
    key_points_col: str


class TrainingConfig(BaseModel):
    resume_ckpt: str | None = None
    device: str
    seed: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    lr: float
    momentum: float
    weight_decay: float
    epochs: int
    eval_interval: int


class ModelConfig(BaseModel):
    num_classes: int
    num_keypoints: int


class Transform(BaseModel):
    name: str
    p: float
    kwargs: dict[str, ty.Any]


class TransformsConfig(BaseModel):
    transforms: list[Transform]


class Config(BaseModel):
    path: PathConfig
    dataset: DatasetConfig
    training: TrainingConfig
    model: ModelConfig
    train_transforms: TransformsConfig
    val_transforms: TransformsConfig
