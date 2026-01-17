import os

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from .postprocessing import decode_keypoints, postprocess_predictions


def visualize_training_samples(
    base_dir: str,
    epoch: int,
    batch: dict[str, torch.Tensor | list[torch.Tensor]],
) -> None:
    images = batch["image"]
    boxes_gt = batch["boxes"]
    kps_gt = batch["key_points"]

    epoch_folder = os.path.join(base_dir, f"epoch_{epoch}")
    os.makedirs(epoch_folder, exist_ok=True)
    for i in range(images.shape[0]):
        img = np.ascontiguousarray(images[i].permute(1, 2, 0).numpy().astype(np.uint8))
        boxes = boxes_gt[i]
        kps = kps_gt[i]
        for box in boxes:
            x1, y1, x2, y2 = box.numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for kp_set in kps:
            for kp in kp_set:
                x, y, vis = kp[:3] if len(kp) >= 3 else (*kp[:2], 1.0)
                if vis > 0:  # Only draw visible keypoints
                    x, y = int(x), int(y)
                    cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(epoch_folder, f"sample_{i}.jpg"), img)


@torch.no_grad()
def visualize_epoch_predictions(
    base_dir: str,
    epoch: int,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor | list[torch.Tensor]],
    device: torch.device,
    conf_threshold: float = 0.5,
    nms_iou_threshold: float = 0.4,
) -> None:
    model.eval()

    images = batch["image"].to(device, non_blocking=True)
    boxes_gt = batch["boxes"]
    kps_gt = batch["key_points"]

    p8_out, p16_out, p32_out = model(images)
    obj_preds, cls_preds, box_preds, kps_preds, grids = postprocess_predictions(
        (p8_out, p16_out, p32_out), (8, 16, 32)
    )

    batch_size = images.shape[0]
    num_samples = 4
    random_indices = np.random.choice(batch_size, size=num_samples, replace=False)

    canvases = []
    for batch_idx in random_indices:
        img = images[batch_idx].cpu()
        img_np = img.permute(1, 2, 0).numpy().astype(np.uint8).copy()

        obj_pred = obj_preds[batch_idx].sigmoid().cpu()
        cls_pred = cls_preds[batch_idx].sigmoid().cpu()
        box_pred = box_preds[batch_idx].cpu()
        kps_pred = kps_preds[batch_idx].cpu()

        grid = grids[batch_idx].cpu()
        kps_decoded = decode_keypoints(kps_pred, grid)

        conf = (obj_pred * cls_pred.squeeze(-1)).sqrt()
        valid_mask = conf > conf_threshold

        if valid_mask.any():
            valid_boxes = box_pred[valid_mask]
            valid_kps = kps_decoded[valid_mask]
            valid_conf = conf[valid_mask]

            if valid_boxes.shape[0] > 1:
                keep_indices = nms(valid_boxes, valid_conf, nms_iou_threshold)
                valid_boxes = valid_boxes[keep_indices]
                valid_kps = valid_kps[keep_indices]
                valid_conf = valid_conf[keep_indices]

            for i in range(valid_boxes.shape[0]):
                x1, y1, x2, y2 = valid_boxes[i].numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                conf_text = f"{valid_conf[i]:.2f}"
                cv2.putText(
                    img_np,
                    conf_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Draw keypoints
                kps = valid_kps[i]
                for kp_idx in range(0, 10, 2):
                    kp_x, kp_y = int(kps[kp_idx]), int(kps[kp_idx + 1])
                    cv2.circle(img_np, (kp_x, kp_y), 3, (0, 255, 0), -1)

        # Draw ground truth (RED)
        if batch_idx < len(boxes_gt):
            gt_boxes = boxes_gt[batch_idx].cpu()
            for i in range(gt_boxes.shape[0]):
                x1, y1, x2, y2 = gt_boxes[i].numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if batch_idx < len(kps_gt):
            gt_kps = kps_gt[batch_idx].cpu()
            for face_idx in range(gt_kps.shape[0]):
                kps = gt_kps[face_idx]
                for kp_idx in range(5):
                    x, y, vis = kps[kp_idx].numpy()
                    if vis > 0:
                        cv2.circle(img_np, (int(x), int(y)), 4, (255, 0, 0), -1)

        canvases.append(img_np)
    first_row = np.hstack((canvases[0], canvases[1]))
    second_row = np.hstack((canvases[2], canvases[3]))
    canvas = np.vstack((first_row, second_row))
    path = os.path.join(base_dir, f"epoch_{epoch}")
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "random_predictions.png")
    cv2.imwrite(path, canvas)
