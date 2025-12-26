import os
import typing as ty
from functools import partial

import cv2
import pandas as pd
import numpy as np
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset, DataLoader

from .utils import get_transforms, transform_list_of_coords_to_array
from .transforms import AugmentationPipeline
from ..config import Config


class DataPoint(ty.TypedDict):
    image: NDArray[np.uint8] | torch.Tensor | cv2.Mat
    boxes: NDArray[np.float32] | torch.Tensor
    box_labels: NDArray[np.float32] | torch.Tensor
    key_points: NDArray[np.float32] | torch.Tensor


class CSVDetectionDataset(Dataset):
    def __init__(
        self,
        config: Config,
        dataframe: pd.DataFrame,
        transforms: AugmentationPipeline | None = None,
    ) -> None:
        self.image_dir = config.path.base_dataset_folder
        self.images = dataframe[config.dataset.image_path_col].apply(
            lambda path: os.path.join(self.image_dir, path)
        ).to_numpy()

        to_float32_array = partial(
            transform_list_of_coords_to_array, dtype=np.float32
        )
        self.boxes = dataframe[config.dataset.boxes_col].apply(
            to_float32_array
        ).to_numpy()
        self.kps = dataframe[config.dataset.key_points_col].apply(
            to_float32_array
        ).to_numpy()
        self.transforms = transforms

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> DataPoint:
        image_path = self.images[index]
        boxes = self.boxes[index]
        kps = self.kps[index]

        # If there are no boxes.
        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            kps = np.zeros((0, 5, 3), dtype=np.float32)
            box_labels = np.zeros((0, 1), dtype=np.float32)
        else:
            # Box is [class_id, top_left_x, top_left_y, bot_right_x, bot_right_y].
            box_labels = boxes[:, 0]
            boxes = boxes[:, 1:]

        # If there are no keypoints.
        if kps.size == 0:
            kps = np.zeros((0, 5, 3), dtype=np.float32)

        # Note that this is not RGB image. Authors didn't convert
        # colorspace and train as is.
        image = cv2.imread(image_path)
        if image is None:
            msg = f"Something wrong with image {image_path}"
            raise ValueError(msg)

        if self.transforms is not None:
            # Key points is [kp1, kp2, kp3, kp4, kp5] where kp_i is [x, y, weight]
            # where weight = 1.0 if kp_i is okay for training and 0.0 otherwise.
            kp_weights = kps[..., 2]
            kps = kps[..., :2]
            image, boxes, kps, box_labels, kp_weights = self.transforms(
                image, boxes, kps, box_labels, kp_weights
            )
            if isinstance(kps, torch.Tensor):
                kps = torch.cat((kps, kp_weights.unsqueeze(dim=-1)), dim=-1)
            elif isinstance(kps, np.ndarray):
                kps = np.dstack((kps, np.expand_dims(kp_weights, axis=-1)))

        return {
            "image": image,
            "boxes": boxes,
            "box_labels": box_labels,
            "key_points": kps,
        }


def detection_collate_fn( data: list[dict[str, ty.Any]]) -> dict[str, ty.Any]:
    images = torch.stack([item["image"] for item in data])
    boxes = [item["boxes"] for item in data]
    kps = [item["key_points"] for item in data]
    box_labels = [item["box_labels"] for item in data]
    return {"image": images, "boxes": boxes, "box_labels": box_labels, "key_points": kps}


def build_dataloaders(
    config: Config, dataframe: pd.DataFrame
) -> dict[str, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(config.training.seed)
    transforms = get_transforms(config)
    dataloaders = {}
    for subset in ("train", "val"):
        subset_df = dataframe.query(f"subset == '{subset}'").reset_index(drop=True)
        dataset = CSVDetectionDataset(config, subset_df, transforms[subset])
        dataloaders[subset] = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=subset=="train",
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            generator=generator,
            collate_fn=detection_collate_fn
        )
    return dataloaders
