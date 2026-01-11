import argparse as ap
import os
import typing as ty
from itertools import islice

import pandas as pd

WIDERFACE_SUBSETS = ("train", "val")


def batched_custom(iterable, n):
    """
    Batches an iterable into chunks of size n.
    Equivalent to itertools.batched in Python 3.12+.
    """
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class AnnotationDict(ty.TypedDict):
    image: str
    boxes: list[int]
    key_points: list[list[int]]


class AnnotationContainer:
    def __init__(
        self, base_dir: str, anno_file_path: str, parse_kps: bool = False
    ) -> None:
        self.parse_kps = parse_kps
        self.base_dir = base_dir
        with open(anno_file_path, "r") as tf:
            self.content = tf.read()
        self.__parse()

    def __parse(self) -> None:
        self.meta = []
        self.images = []
        self.labels = []

        samples_as_text = self.content.split("# ")[
            1:
        ]  # We skip 1 element since it's an empty string.
        samples_as_text = [sample.strip().split("\n") for sample in samples_as_text]
        for sample in samples_as_text:
            image_meta, *boxes_str = sample

            name, height, width = image_meta.split(" ")
            image_path = os.path.join(
                *self.base_dir.split(os.path.sep), *name.split("/")
            )  # This should be correct for Windows and Unix.
            self.images.append(image_path)
            self.meta.append((height, width))

            labels = {"boxes": [], "key_points": []}
            for i, point_set in enumerate(boxes_str):
                coords = list(map(float, point_set.strip().split(" ")))

                # Box is a first 4 coordinates in top_left_x, top_left_y, bottom_right_x, bottom_right_y format.
                box = list(map(int, coords[:4]))

                # Artificallly add class label to box.
                box = [0, *box]
                if any(coord < 0 for coord in box):
                    msg = f"Image {image_path} has box with negative coords: {box}"
                    raise ValueError(msg)
                labels["boxes"].append(box)

                # Key points are the rest points. It should be 5 in total, 3 components each (x, y, visibility flag).
                if self.parse_kps:
                    kps = []
                    for point in batched_custom(coords[4:], 3):
                        if all(coord == -1 for coord in point):
                            kps.append([-1.0, -1.0, 0.0])
                        else:
                            point = list(point)
                            point[-1] = 1.0
                            kps.append(point)
                    if len(kps) != 5:
                        msg = f"Image {image_path} has more or less than 5 kps: {kps}"
                        raise ValueError(msg)
                    labels["key_points"].append(kps)

            self.labels.append(labels)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> AnnotationDict:
        image = self.images[index]
        label = self.labels[index]
        key_points = label["key_points"]
        boxes = label["boxes"]
        return {"image": image, "boxes": boxes, "key_points": key_points}

    def __iter__(self):
        for image_path, label in zip(self.images, self.labels):
            yield {"image": image_path, **label}


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument(
        "--data_path", "-d", required=True, type=str, help="Path to WiderFace"
    )
    args = parser.parse_args()

    datasets = []
    for subset in WIDERFACE_SUBSETS:
        anno_file_path = os.path.join(args.data_path, "labelv2", subset, "labelv2.txt")
        image_dir = os.path.join(
            args.data_path, f"WIDER_{subset}", f"WIDER_{subset}", "images"
        )
        container = AnnotationContainer(image_dir, anno_file_path, subset == "train")
        dataframe = pd.DataFrame(data=container)
        dataframe["subset"] = subset
        dataframe["source"] = "widerface"
        datasets.append(dataframe)
    dataset_df = pd.concat(datasets)
    dataset_df_path = os.path.join(args.data_path, "widerface_main_chck.csv")
    dataset_df.to_csv(dataset_df_path, index=False)
