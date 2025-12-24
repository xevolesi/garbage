import typing as ty

import cv2
import pandas as pd
import albumentations as album

from ..config import Config
from .transforms import RandomSquareCrop


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
            bbox_params=album.BboxParams(format="pascal_voc"),
            keypoint_params=album.KeypointParams(format="xy", remove_invisible=False), 
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
            bbox_params=album.BboxParams(format="pascal_voc"),
            keypoint_params=album.KeypointParams(format="xy", remove_invisible=False), 
        )
    else:
        msg = f"Invalid `subset` argument. Expected one of ('train', 'val'), but got {subset}"
        raise ValueError(msg)