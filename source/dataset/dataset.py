import os
import typing as ty

import cv2
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import albumentations as album

import torch
from torch.utils.data import Dataset, DataLoader

from .utils import get_transforms


NPDType = ty.TypeVar("NPDType", bound=np.generic)
NPImageT = ty.Annotated[NDArray[NPDType], ty.Literal[ty.Any, ty.Any, 3]]
NPBoxT = ty.Annotated[NDArray[NPDType], ty.Literal[ty.Any, 4]]
NPKeyPointsT = ty.Annotated[NDArray[NPDType], ty.Literal[ty.Any, 5, 2]]
ImageT: ty.TypeAlias = NPImageT | torch.Tensor


class DataPoint(ty.TypedDict):
    image: ImageT
    boxes: NPBoxT
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
        self.boxes = dataframe[boxes_col].apply(np.array).to_numpy()
        self.kps = dataframe[key_points_col].apply(np.array).to_numpy()
        self.transforms = transforms

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> DataPoint:
        image_path = self.images[index]
        boxes = self.boxes[index]
        kps = self.kps[index]
        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            kps = np.zeros((0, 5, 2), dtype=np.float32)
        if kps.size == 0:
            kps = np.zeros((0, 5, 2), dtype=np.float32)
        image = cv2.imread(image_path)
        if image is None:
            msg = f"Something wrong with image {image_path}"
            raise ValueError(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Okay, here we have boxes and keypoints as a NumPy arrays of shape
        # (N faces on image, 4 coordinates in box) for box and
        # (N faces on image, M points, 2 coordinates) for key points. I noticed
        # that there are boxes with area less than 100 pixels. I want to remove
        # them from dataset, but keep sample. Just remove particular box from
        # the image. We also need to remove corresponding key points.
        if boxes.size > 0:
            box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            boxes = boxes[box_areas > 100] # Nx4
            if kps.size > 0:
                kps = kps[box_areas > 100] # Nx5x2

        # Some key points may be not usefull for training in some reasons.
        # Such keypoints are [-1, ...] array, so if all 5 key points are
        # useless you get [-1, -1, -1, -1, -1] array as a label. Unfortunately
        # albumentations doesn't handle them as is, so we need to do some funny
        # thing: mask out invalid key points, pass masked valid keypoints to
        # albumentation transformations and only then add back invalid keypoints
        # to filter them later during target creation process.
        # Note that we trhow away all 5 keypoints for single face, so we throw
        # away whole face. So, there might be box without key points.
        
        valid_kps_idx = [idx for idx, _ in enumerate(kps)]
        valid_kps = kps[valid_kps_idx]
        if kps.size > 0:
            valid_kps_idx = [idx for idx, face_kps in enumerate(kps) if (face_kps != -1).all()]
            valid_kps = kps[valid_kps_idx]

        # Some coordinates may exceed image dimensions. So, we need to handle
        # such situations to allow albumentations works correctly. In doing so
        # we need to clip boxes' and key points' coordinates to match image
        # dimensions.
        if boxes.size > 0:
            boxes[:, ::2] = np.clip(boxes[:, ::2], 1.0, image.shape[1] - 1)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 1.0, image.shape[0] - 1)
        if valid_kps.size > 0:
            valid_kps[:, :, ::2] = np.clip(valid_kps[:, :, ::2], 1.0, image.shape[1] - 1)
            valid_kps[:, :, 1::2] = np.clip(valid_kps[:, :, 1::2], 1.0, image.shape[0] - 1)

        if self.transforms is not None:
            augmented = self.transforms(
                image=image,
                bboxes=boxes,

                # We need such shape since albumentations can't work with
                # "batch of faces of key points", e. g. it's not working
                # with (N faces, M points, 2) shape.
                keypoints=valid_kps.reshape(-1, 2),
            )
            image = augmented["image"]
            boxes = augmented["bboxes"]

            # We need to put back invalid points and reshape valid key points
            # back to shape (N faces, M points, 2) to be able to distinguish
            # between parent faces during target creation procedure.
            # Let's reshape back first.
            valid_kps = augmented["keypoints"].reshape(-1, 5, 2)
            kps = np.full(kps.shape, -1)
            kps[valid_kps_idx] = valid_kps

        return {
            "image": image,
            "boxes": boxes,
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
        image_dir = os.path.join(base_data_path, f"WIDER_{subset}", f"WIDER_{subset}", "images")
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
