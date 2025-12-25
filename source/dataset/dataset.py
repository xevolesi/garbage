import os
import typing as ty
from functools import partial

import cv2
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import albumentations as album

import torch
from torch.utils.data import Dataset, DataLoader

from .utils import get_transforms, transform_list_of_coords_to_array


NPDType = ty.TypeVar("NPDType", bound=np.generic)
NPImageT = ty.Annotated[NDArray[NPDType], ty.Literal[ty.Any, ty.Any, 3]]
NPBoxT = ty.Annotated[NDArray[NPDType], ty.Literal[ty.Any, 4]]
NPKeyPointsT = ty.Annotated[NDArray[NPDType], ty.Literal[ty.Any, 5, 2]]
ImageT: ty.TypeAlias = NPImageT | torch.Tensor


class DataPoint(ty.TypedDict):
    image: ImageT
    boxes: NPBoxT
    box_labels: NDArray
    key_points: NPKeyPointsT


class CSVDetectionDataset(Dataset):
    face_classes: ty.ClassVar[list[str]] = ["face"]
    point_classes: ty.ClassVar[list[str]] = ["left_eye", "right_eye", "nose", "left_lips_corner", "right_lips_corner"]

    def __init__(
        self,
        image_dir: str,
        dataframe: pd.DataFrame,
        image_path_col: str,
        boxes_col: str,
        key_points_col: str,
        transforms: album.Compose | None = None,
    ) -> None:
        self.image_dir = image_dir
        self.images = dataframe[image_path_col].apply(lambda path: os.path.join(self.image_dir, path)).to_numpy()

        to_float32_array = partial(transform_list_of_coords_to_array, dtype=np.float32)
        self.boxes = dataframe[boxes_col].apply(to_float32_array).to_numpy()
        self.kps = dataframe[key_points_col].apply(to_float32_array).to_numpy()
        self.transforms = transforms

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> DataPoint:
        image_path = self.images[index]
        boxes = self.boxes[index]
        kps = self.kps[index]

        # Box is [class_id, top_left_x, top_left_y, bot_right_x, bot_right_y].
        # Key points is [kp1, kp2, kp3, kp4, kp5] where kp_i is [x, y, weight]
        # where weight = 1.0 if kp_i is okay for training and 0.0 otherwise.
        box_labels = boxes[:, 0]
        boxes = boxes[:, 1:]

        # If there are no boxes.
        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            kps = np.zeros((0, 5, 3), dtype=np.float32)
            box_labels = np.zeros((0, 1), dtype=np.float32)

        # If there are no keypoints.
        if kps.size == 0:
            kps = np.zeros((0, 5, 3), dtype=np.float32)

        # Note that this is not RGB image. Authors didn't convert
        # colorspace and train as is.
        image = cv2.imread(image_path)
        if image is None:
            msg = f"Something wrong with image {image_path}"
            raise ValueError(msg)

        kp_weights = kps[..., 2].reshape(-1, 1)
        if self.transforms is not None:
            augmented = self.transforms(
                image=image,
                bboxes=boxes,
                box_labels=box_labels,

                # We need such shape since albumentations can't work with
                # "batch of faces of key points", e. g. it's not working
                # with (N faces, M points, 2) shape.
                keypoints=kps[..., :2].reshape(-1, 2),
                kp_indices=np.arange(kps[..., :2].shape[0] * 5),
            )
            image = augmented["image"]
            boxes = augmented["bboxes"]
            box_labels = augmented["box_labels"]

            # Duringa augmentations some key points may change it's index.
            # For example in case of horizontal flipping left eye point will
            # be right point and visa versa. So, to track such changes and
            # reflect it in kp_weights we need to pass indices into albu
            # compose object and later reindex kp_weight to match augmented
            # key points.
            kps = augmented["keypoints"]
            kp_indices = augmented["kp_indices"]
            kps = np.hstack((kps, kp_weights[kp_indices]))
            kps = kps.reshape(-1, 5, 3)

        return {
            "image": image,
            "boxes": boxes,
            "box_labels": box_labels,
            "key_points": kps,
        }


def detection_collate_fn( data: list[dict[str, ty.Any]]) -> dict[str, ty.Any]:
    images = torch.stack([item["image"] for item in data])
    boxes = [torch.from_numpy(item["boxes"]) for item in data]
    kps = [torch.from_numpy(item["key_points"]) for item in data]
    return {"image": images, "boxes": boxes, "key_points": kps}


def build_dataloaders(
    base_data_path: str, dataframe: pd.DataFrame, device: torch.device
) -> dict[str, DataLoader]:
    generator = torch.Generator(device=device)
    dataloaders = {}
    for subset in ("train", "val"):
        image_dir = os.path.join(base_data_path, f"WIDER_train", f"WIDER_train", "images")
        subset_df = dataframe.query(f"subset == '{subset}'").reset_index(drop=True)
        transforms = get_transforms(subset=subset)
        dataset = CSVDetectionDataset(
            image_dir, subset_df, "image", "boxes", "key_points", transforms
        )
        dataloaders[subset] = DataLoader(
            dataset,
            batch_size=32,
            shuffle=subset=="train",
            num_workers=6,
            generator=generator,
            collate_fn=detection_collate_fn
        )
    return dataloaders
