import typing as ty

import cv2
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import albumentations as album

from ..config import Config
from .transforms import RandomSquareCrop


def transform_list_of_coords_to_array(list_of_coords: str, dtype=np.float32) -> NDArray:
    """
    Transform '[[x1, y1, x2, y2], ...[x1, y1, x2, y2]]` to NumPy array.
    Note that function uses `eval` python function so it's not safe.

    Args:
        list_of_coords: String representation of bounding box list or point list;
        dtype: Resulting NumPy array data type.
    
    Returns:
        Transformed boudning boxes or points as numpy array.
        Resulting shape is (N_items_in_list, M_coords_in_item).
    """
    arr = np.array(eval(list_of_coords))
    return arr.astype(dtype)


def train_val_test_split(config:Config, dataframe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_df = dataframe.query(f"fold in {config.dataset.train_folds}").reset_index(drop=True)
    val_df = dataframe.query(f"fold in {config.dataset.train_folds}").reset_index(drop=True)
    return {"train": train_df, "val_df": val_df}


def get_transforms(subset: ty.Literal["val", "train"]) -> album.Compose:
    if subset == "val":
        return album.Compose(
            [
                album.LongestMaxSize(max_size=640, interpolation=cv2.INTER_AREA, p=1.0),
                album.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
                album.ToFloat(max_value=255, p=1.0),
                album.pytorch.ToTensorV2(),
            ],
            bbox_params=album.BboxParams(format="pascal_voc", label_fields=["box_labels"]),
            keypoint_params=album.KeypointParams(format="xy", remove_invisible=False, label_fields=["kp_indices"]), 
        )
    elif subset == "train":
        return album.Compose(
            [
                RandomSquareCrop(crop_choice=[0.5, 0.7, 0.9, 1.1, 1.3, 1.5], p=1.0),
                album.Resize(height=640, width=640, interpolation=cv2.INTER_AREA),
                album.HorizontalFlip(p=0.5),
                album.ToFloat(max_value=255, p=1.0),
                album.pytorch.ToTensorV2(),
            ],
            bbox_params=album.BboxParams(format="pascal_voc", label_fields=["box_labels"]),
            keypoint_params=album.KeypointParams(format="xy", remove_invisible=False, label_fields=["kp_indices"]), 
        )
    else:
        msg = f"Invalid `subset` argument. Expected one of ('train', 'val'), but got {subset}"
        raise ValueError(msg)