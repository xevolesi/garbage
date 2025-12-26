import os
import typing as ty

import torch


def draw_random_images_from_batch(
    epoch: int, base_dir: str, batch: dict[str, ty.Any]
) -> None:
    images = batch["image"]
    boxes = batch["boxes"]
    kps = batch["key_points"]
    