import typing as ty

import cv2
import albumentations as album


def get_transforms(subset: ty.Literal["val", "train"]) -> album.Compose:
    if subset == "val":
        return album.Compose(
            [
                album.Resize(height=640, width=640, interpolation=cv2.INTER_AREA),
                album.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), # Should cast image into [-1, 1].
                album.pytorch.ToTensorV2(),
            ],
            bbox_params=album.BboxParams(format="pascal_voc"),
            keypoint_params=album.KeypointParams(format="xy"), 
        )
    elif subset == "train":
        return album.Compose(
            [
                album.Resize(height=640, width=640, interpolation=cv2.INTER_AREA),
                album.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), # Should cast image into [-1, 1].
                album.pytorch.ToTensorV2(),
            ],
            bbox_params=album.BboxParams(format="pascal_voc"),
            keypoint_params=album.KeypointParams(format="xy"), 
        )
    else:
        msg = f"Invalid `subset` argument. Expected one of ('train', 'val'), but got {subset}"
        raise ValueError(msg)