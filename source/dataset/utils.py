import numpy as np
from numpy.typing import NDArray

from ..config import Config
from .transforms import AugmentationPipeline


def transform_list_of_coords_to_array(list_of_coords: str, dtype=np.float32) -> NDArray:
    arr = np.array(eval(list_of_coords))
    return arr.astype(dtype)


def get_transforms(config: Config) -> dict[str, AugmentationPipeline]:
    return {
        "train": AugmentationPipeline(config.train_transforms),
        "val": AugmentationPipeline(config.val_transforms)
    }
