import random
from typing import Tuple, Optional

import cv2
import torch
import numpy as np
from numpy.typing import NDArray

from ..config import TransformsConfig


class AugmentationPipeline:
    """
    Pipeline for face detection augmentations.

    Example:
        pipeline = AugmentationPipeline([
            (horizontal_flip, 0.5, {}),
            (coarse_dropout, 0.3, {'max_holes': 8}),
            (random_brightness_contrast, 0.5, {}),
        ])

        aug_img, aug_boxes, aug_kps, aug_labels, aug_weights = pipeline(
            image, boxes, keypoints, labels, keypoints_weights
        )
    """

    def __init__(self, transforms_cfg: TransformsConfig) -> None:
        """
        Args:
            transforms: List of (function, probability, kwargs)
                - function: augmentation function
                - probability: probability of applying this transform (0.0 to 1.0)
                - kwargs: additional keyword arguments for the function
        """
        self.transform_cfg = transforms_cfg

    def __call__(
        self,
        image: NDArray[np.uint8],
        boxes: NDArray[np.float32],
        keypoints: NDArray[np.float32],
        box_labels: NDArray[np.int64],
        keypoints_weights: NDArray[np.float32],
    ) -> tuple[
        NDArray | torch.Tensor,
        NDArray | torch.Tensor,
        NDArray | torch.Tensor,
        NDArray | torch.Tensor,
        NDArray | torch.Tensor,
    ]:
        for transform_model in self.transform_cfg.transforms:
            p = transform_model.p
            kwargs = transform_model.kwargs
            method = getattr(self, transform_model.name)
            if np.random.rand() < p:
                image, boxes, keypoints, box_labels, keypoints_weights = method(
                    image, boxes, keypoints, box_labels, keypoints_weights, **kwargs
                )
        return image, boxes, keypoints, box_labels, keypoints_weights

    def _crop_image_with_padding(
        self,
        image: NDArray[np.uint8],
        patch: NDArray[np.int32],
        crop_size: Tuple[int, int],
        fill_value: int,
    ) -> NDArray[np.uint8]:
        """
        Crop image with padding if crop extends beyond boundaries.

        Args:
            image: (H, W, 3)
            patch: [left, top, right, bottom] - can be negative
            crop_size: (height, width)
            fill_value: padding fill value

        Returns:
            cropped_image: (crop_height, crop_width, 3)
        """
        h, w = image.shape[:2]
        ch, cw = crop_size

        # Create output image with fill value
        if isinstance(fill_value, (int, float)):
            result = np.ones((ch, cw, image.shape[2]), dtype=image.dtype) * fill_value
        else:
            result = np.ones((ch, cw, image.shape[2]), dtype=image.dtype)
            result[:, :] = fill_value

        # Calculate region to copy from (source image)
        src_x1 = max(0, patch[0])
        src_y1 = max(0, patch[1])
        src_x2 = min(w, patch[2])
        src_y2 = min(h, patch[3])

        # Calculate region to copy to (result image)
        dst_x1 = max(0, -patch[0])
        dst_y1 = max(0, -patch[1])
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        # Copy data
        result[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

        return result

    def random_square_crop(
        self,
        image: NDArray[np.uint8],
        boxes: NDArray[np.float32],
        keypoints: NDArray[np.float32],
        box_labels: NDArray[np.int64],
        keypoints_weights: NDArray[np.float32],
        crop_ratio_range: Optional[Tuple[float, float]] = None,
        crop_choice: Optional[list] = None,
        bbox_clip_border: bool = True,
        fill_value: int = 128,
        max_retry: int = 250,
        scale_retry_limit: int = 10,
    ) -> Tuple[NDArray, ...]:
        """
        Random square crop with padding and retry mechanism (YuNet style).

        Strategy:
        1. Select scale (from range or choice)
        2. Generate random crop position (can be negative for padding)
        3. Check that all bbox centers are inside crop
        4. If not suitable - retry up to 250 times
        5. If not found after 250 attempts - increase scale and retry
        6. After scale_retry_limit - use fallback (center crop)

        Args:
            image: (H, W, 3)
            boxes: (n_faces, 4) - [x1, y1, x2, y2]
            keypoints: (n_faces, 5, 2)
            box_labels: (n_faces,)
            keypoints_weights: (n_faces, 5)
            crop_ratio_range: Range for crop scale, e.g. (0.5, 2.0)
            crop_choice: List of crop scales, e.g. [0.5, 1.0, 1.5, 2.0]
            bbox_clip_border: Whether to clip bboxes/keypoints to crop border
            fill_value: Padding fill value (RGB or int)
            max_retry: Max iterations per scale
            scale_retry_limit: Max scale retry attempts

        Returns:
            cropped (image, boxes, keypoints, box_labels, keypoints_weights)
        """
        assert (crop_ratio_range is None) ^ (crop_choice is None), (
            "Either crop_ratio_range or crop_choice must be specified, but not both"
        )

        h, w = image.shape[:2]
        if crop_ratio_range is not None:
            crop_ratio_min, crop_ratio_max = crop_ratio_range
            max_scale = crop_ratio_max
        else:
            max_scale = np.amax(crop_choice)

        # Retry mechanism to find suitable crop
        scale_retry = 0
        scale = None
        patch = None

        while True:
            scale_retry += 1

            # Determine scale for current iteration
            if scale_retry == 1 or max_scale > 1.0:
                if crop_ratio_range is not None:
                    scale = np.random.uniform(crop_ratio_min, crop_ratio_max)
                else:
                    scale = np.random.choice(crop_choice)
            else:
                # Increase scale if previous attempt failed
                scale = scale * 1.2

            # Try to find suitable crop with current scale
            for _ in range(max_retry):
                short_side = min(w, h)
                cw = int(scale * short_side)
                ch = cw  # square crop

                # Calculate crop position (can be negative for padding)
                if w == cw:
                    left = 0
                elif w > cw:
                    left = random.randint(0, w - cw)
                else:
                    left = random.randint(w - cw, 0)

                if h == ch:
                    top = 0
                elif h > ch:
                    top = random.randint(0, h - ch)
                else:
                    top = random.randint(h - ch, 0)

                patch = np.array([left, top, left + cw, top + ch])

                # Check that bbox centers are inside crop
                if len(boxes) > 0:
                    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                    mask = (
                        (centers[:, 0] > patch[0])
                        & (centers[:, 1] > patch[1])
                        & (centers[:, 0] < patch[2])
                        & (centers[:, 1] < patch[3])
                    )

                    if not mask.any():
                        continue

                # Successfully found suitable crop
                break
            else:
                # Not found after max_retry iterations
                if scale_retry > scale_retry_limit:
                    # Fallback - center crop
                    short_side = min(w, h)
                    cw = int(scale * short_side)
                    ch = cw
                    left = (w - cw) // 2
                    top = (h - ch) // 2
                    patch = np.array([left, top, left + cw, top + ch])
                    break
                # Retry with increased scale
                continue

            # Found suitable crop
            break

        # Apply crop to image
        cropped_image = self._crop_image_with_padding(
            image, patch, (ch, cw), fill_value
        )

        # Apply crop to boxes and keypoints
        cropped_boxes = []
        cropped_keypoints = []
        cropped_labels = []
        cropped_weights = []

        for i in range(len(boxes)):
            bbox = boxes[i]
            kps = keypoints[i]
            label = box_labels[i]
            weights = keypoints_weights[i]

            # Check bbox center
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            if not (patch[0] < center_x < patch[2] and patch[1] < center_y < patch[3]):
                # Center outside crop - skip this bbox
                continue

            # Clip bbox to crop borders if needed
            x_min, y_min, x_max, y_max = bbox
            if bbox_clip_border:
                x_min = np.clip(x_min, patch[0], patch[2])
                y_min = np.clip(y_min, patch[1], patch[3])
                x_max = np.clip(x_max, patch[0], patch[2])
                y_max = np.clip(y_max, patch[1], patch[3])

            # Shift relative to crop
            x_min -= patch[0]
            y_min -= patch[1]
            x_max -= patch[0]
            y_max -= patch[1]

            cropped_boxes.append([x_min, y_min, x_max, y_max])

            # Process keypoints
            kps_new = kps.copy()
            weights_new = weights.copy()

            for kp_idx in range(5):
                x, y = kps[kp_idx]

                # Check that keypoint is inside crop
                if bbox_clip_border:
                    x = np.clip(x, patch[0], patch[2])
                    y = np.clip(y, patch[1], patch[3])

                # If keypoint outside crop - zero out weight
                if x < patch[0] or x > patch[2] or y < patch[1] or y > patch[3]:
                    weights_new[kp_idx] = 0.0

                # Shift relative to crop
                kps_new[kp_idx] = [x - patch[0], y - patch[1]]

            cropped_keypoints.append(kps_new)
            cropped_labels.append(label)
            cropped_weights.append(weights_new)

        # Convert to numpy arrays
        if len(cropped_boxes) > 0:
            cropped_boxes = np.array(cropped_boxes, dtype=np.float32)
            cropped_keypoints = np.array(cropped_keypoints, dtype=np.float32)
            cropped_labels = np.array(cropped_labels, dtype=box_labels.dtype)
            cropped_weights = np.array(cropped_weights, dtype=np.float32)
        else:
            # Empty arrays with correct dimensions
            cropped_boxes = np.empty((0, 4), dtype=np.float32)
            cropped_keypoints = np.empty((0, 5, 2), dtype=np.float32)
            cropped_labels = np.empty((0,), dtype=box_labels.dtype)
            cropped_weights = np.empty((0, 5), dtype=np.float32)

        return (
            cropped_image,
            cropped_boxes,
            cropped_keypoints,
            cropped_labels,
            cropped_weights,
        )

    def horizontal_flip(
        self,
        image: NDArray[np.uint8],
        boxes: NDArray[np.float32],
        keypoints: NDArray[np.float32],
        box_labels: NDArray[np.float32],
        keypoints_weights: NDArray[np.float32],
    ) -> tuple[NDArray, ...]:
        flip_order = [1, 0, 2, 4, 3]
        h, w = image.shape[:2]

        image = np.fliplr(image)
        boxes = boxes.copy()
        keypoints = keypoints[:, flip_order, :].copy()
        keypoints_weights = keypoints_weights[:, flip_order].copy()

        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        keypoints[:, :, 0] = w - keypoints[:, :, 0]

        return image, boxes, keypoints, box_labels, keypoints_weights

    def letter_box(
        self,
        image: NDArray[np.uint8],
        boxes: NDArray[np.float32],
        keypoints: NDArray[np.float32],
        box_labels: NDArray[np.int64],
        keypoints_weights: NDArray[np.float32],
    ) -> tuple[NDArray, ...]:
        """
        Letterbox padding to make image square.
        Pads shorter side to match longer side with black borders.

        Args:
            image: (H, W, 3)
            boxes: (n_faces, 4) - [x1, y1, x2, y2]
            keypoints: (n_faces, 5, 2)
            box_labels: (n_faces,)
            keypoints_weights: (n_faces, 5)

        Returns:
            letterboxed (image, boxes, keypoints, box_labels, keypoints_weights)
        """
        h, w = image.shape[:2]
        max_size = max(h, w)

        # Calculate padding (handle odd sizes correctly)
        pad_w = (max_size - w) // 2
        pad_h = (max_size - h) // 2
        pad_w_right = max_size - w - pad_w  # remaining padding for right
        pad_h_bottom = max_size - h - pad_h  # remaining padding for bottom

        # Apply padding
        image = cv2.copyMakeBorder(
            image,
            pad_h,
            pad_h_bottom,  # top, bottom
            pad_w,
            pad_w_right,  # left, right
            cv2.BORDER_CONSTANT,
            value=0,
        )

        # Shift boxes and keypoints (copy before modification)
        boxes = boxes.copy()
        boxes[:, [0, 2]] += pad_w  # x coordinates
        boxes[:, [1, 3]] += pad_h  # y coordinates

        keypoints = keypoints.copy()
        keypoints[:, :, 0] += pad_w  # x coordinates
        keypoints[:, :, 1] += pad_h  # y coordinates

        return image, boxes, keypoints, box_labels, keypoints_weights

    def resize(
        self,
        image: NDArray[np.uint8],
        boxes: NDArray[np.float32],
        keypoints: NDArray[np.float32],
        box_labels: NDArray[np.int64],
        keypoints_weights: NDArray[np.float32],
        target_size: Tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR,
    ) -> tuple[NDArray, ...]:
        """
        Resize image and scale boxes/keypoints accordingly.

        Args:
            image: (H, W, 3)
            boxes: (n_faces, 4) - [x1, y1, x2, y2]
            keypoints: (n_faces, 5, 2)
            box_labels: (n_faces,)
            keypoints_weights: (n_faces, 5)
            target_size: (height, width) - target image size
            interpolation: cv2 interpolation method (INTER_LINEAR, INTER_AREA, etc.)

        Returns:
            resized (image, boxes, keypoints, box_labels, keypoints_weights)
        """
        h, w = image.shape[:2]
        target_h, target_w = target_size

        # If already target size, return early
        if h == target_h and w == target_w:
            return image, boxes, keypoints, box_labels, keypoints_weights

        # Calculate scale factors
        scale_y = target_h / h
        scale_x = target_w / w

        # Resize image
        image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)

        # Scale boxes
        boxes = boxes.copy()
        boxes[:, [0, 2]] *= scale_x  # x coordinates
        boxes[:, [1, 3]] *= scale_y  # y coordinates

        # Scale keypoints
        keypoints = keypoints.copy()
        keypoints[:, :, 0] *= scale_x  # x coordinates
        keypoints[:, :, 1] *= scale_y  # y coordinates

        return image, boxes, keypoints, box_labels, keypoints_weights

    def to_float32_tensor(
        self,
        image: NDArray[np.uint8],
        boxes: NDArray[np.float32],
        keypoints: NDArray[np.float32],
        box_labels: NDArray[np.int64],
        keypoints_weights: NDArray[np.float32],
    ) -> tuple[torch.Tensor, ...]:
        image = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .to(torch.float32)
        )
        boxes = torch.from_numpy(boxes).to(torch.float32)
        keypoints = torch.from_numpy(keypoints).to(torch.float32)
        box_labels = torch.from_numpy(box_labels).to(torch.float32)
        keypoints_weights = torch.from_numpy(keypoints_weights).to(torch.float32)
        return image, boxes, keypoints, box_labels, keypoints_weights
