"""
WiderFace Evaluation Script for YuNet

Reproduces the official WiderFace evaluation protocol from libfacedetection.train.
Evaluates trained YuNet models on easy/medium/hard difficulty levels.

Original evaluation code by wondervictor (tianhengcheng@gmail.com).

Please note that this evaluation script is not the same as the one in libfacedetection.train.
The difference is that this script performs the following preprocessing:

- Letterbox to square (using max dimension)
- Pad to nearest multiple of 32
- No resizing (original scale preserved)

This is almost the same as the preprocessing in libfacedetection.train,
but with the difference that this script does letterbox padding.

Major part of the code was borrowed from libfacedetection.train https://github.com/ShiqiYu/libfacedetection.train
"""

from __future__ import annotations

import argparse
import datetime
import os
from multiprocessing import Pool

import cv2
import numpy as np
import torch
import torchvision
from scipy.io import loadmat
from tqdm import tqdm

from source.models import YuNet
from source.postprocessing import postprocess_predictions


def bbox_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute IoU between multiple boxes (a) and a single box (b).

    Args:
        a: Array of boxes with shape (N, 4) in [x1, y1, x2, y2] format
        b: Single box with shape (4,) in [x1, y1, x2, y2] format

    Returns:
        IoU values with shape (N,)
    """
    x1 = np.maximum(a[:, 0], b[0])
    y1 = np.maximum(a[:, 1], b[1])
    x2 = np.minimum(a[:, 2], b[2])
    y2 = np.minimum(a[:, 3], b[3])
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    inter = w * h
    aarea = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    o = inter / (aarea + barea - inter)
    o[w <= 0] = 0
    o[h <= 0] = 0
    return o


def get_gt_boxes(gt_dir: str) -> tuple:
    """
    Load ground truth boxes from .mat files.

    Args:
        gt_dir: Directory containing wider_face_val.mat, wider_easy_val.mat,
                wider_medium_val.mat, wider_hard_val.mat

    Returns:
        Tuple of (facebox_list, event_list, file_list,
                  hard_gt_list, medium_gt_list, easy_gt_list)
    """
    gt_mat = loadmat(os.path.join(gt_dir, "wider_face_val.mat"))
    hard_mat = loadmat(os.path.join(gt_dir, "wider_hard_val.mat"))
    medium_mat = loadmat(os.path.join(gt_dir, "wider_medium_val.mat"))
    easy_mat = loadmat(os.path.join(gt_dir, "wider_easy_val.mat"))

    facebox_list = gt_mat["face_bbx_list"]
    event_list = gt_mat["event_list"]
    file_list = gt_mat["file_list"]

    hard_gt_list = hard_mat["gt_list"]
    medium_gt_list = medium_mat["gt_list"]
    easy_gt_list = easy_mat["gt_list"]

    return (
        facebox_list,
        event_list,
        file_list,
        hard_gt_list,
        medium_gt_list,
        easy_gt_list,
    )


def norm_score(pred: dict) -> dict:
    """
    Normalize prediction scores to [0, 1] range.

    Args:
        pred: Dict of predictions {event_name: {img_name: np.array of [x,y,w,h,s]}}

    Returns:
        Same dict with normalized scores
    """
    max_score = -1
    min_score = 2

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    if diff == 0:
        diff = 1.0  # Avoid division by zero

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score).astype(np.float64) / diff
    return pred


def image_eval(
    pred: np.ndarray,
    gt: np.ndarray,
    ignore: np.ndarray,
    iou_thresh: float,
    pool: Pool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single image evaluation.

    Args:
        pred: Predictions with shape (N, 5) in [x, y, w, h, score] format
        gt: Ground truth boxes with shape (M, 4) in [x, y, w, h] format
        ignore: Ignore flags with shape (M,)
        iou_thresh: IoU threshold for matching
        pool: Multiprocessing pool

    Returns:
        Tuple of (pred_recall, proposal_list)
    """
    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    # Convert from xywh to xyxy
    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    gt_overlap_list = pool.starmap(
        bbox_overlap,
        zip([_gt] * _pred.shape[0], [_pred[h] for h in range(_pred.shape[0])]),
    )

    for h in range(_pred.shape[0]):
        gt_overlap = gt_overlap_list[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()

        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)

    return pred_recall, proposal_list


def img_pr_info(
    thresh_num: int,
    pred_info: np.ndarray,
    proposal_list: np.ndarray,
    pred_recall: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-image PR info across thresholds.

    Args:
        thresh_num: Number of threshold steps
        pred_info: Predictions with scores
        proposal_list: Valid proposal flags
        pred_recall: Recall counts

    Returns:
        Tuple of (pr_info, fp)
    """
    pr_info = np.zeros((thresh_num, 2)).astype("float")
    fp = np.zeros((pred_info.shape[0],), dtype=np.int32)

    for t in range(thresh_num):
        thresh = 1 - (t + 1) / thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[: r_index + 1] == 1)[0]
            pr_info[t, 0] = len(p_index)  # valid pred number
            pr_info[t, 1] = pred_recall[r_index]  # valid gt number

            if (
                t > 0
                and pr_info[t, 0] > pr_info[t - 1, 0]
                and pr_info[t, 1] == pr_info[t - 1, 1]
            ):
                fp[r_index] = 1
    return pr_info, fp


def dataset_pr_info(
    thresh_num: int, pr_curve: np.ndarray, count_face: int
) -> np.ndarray:
    """
    Compute dataset-level PR curve.

    Args:
        thresh_num: Number of threshold steps
        pr_curve: Accumulated PR curve
        count_face: Total face count

    Returns:
        Normalized PR curve with shape (thresh_num, 2)
    """
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        if pr_curve[i, 0] > 0:
            _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        else:
            _pr_curve[i, 0] = 0
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """
    Compute VOC-style Average Precision.

    Args:
        rec: Recall values
        prec: Precision values

    Returns:
        Average Precision value
    """
    # Append sentinel values at the end
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Find points where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum (delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


def wider_evaluation(
    pred: dict, gt_path: str, iou_thresh: float = 0.5, num_workers: int = 8
) -> list[float]:
    """
    Run WiderFace evaluation.

    Args:
        pred: Dict of predictions {event_name: {img_name: np.array of [x,y,w,h,s]}}
        gt_path: Path to ground truth directory
        iou_thresh: IoU threshold for matching
        num_workers: Number of worker processes

    Returns:
        List of AP values [easy_ap, medium_ap, hard_ap]
    """
    pred = norm_score(pred)
    thresh_num = 1000

    (
        facebox_list,
        event_list,
        file_list,
        hard_gt_list,
        medium_gt_list,
        easy_gt_list,
    ) = get_gt_boxes(gt_path)

    event_num = len(event_list)
    settings = ["easy", "medium", "hard"]
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]

    pool = Pool(num_workers)
    aps = [-1.0, -1.0, -1.0]

    print("")
    for setting_id in range(3):
        ta = datetime.datetime.now()
        iou_th = iou_thresh
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype("float")

        for i in range(event_num):
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            pred_list = pred.get(event_name, {})
            sub_gt_list = gt_list[i][0]
            gt_bbx_list = facebox_list[i][0]

            for j in range(len(img_list)):
                img_name = str(img_list[j][0][0])
                pred_info = pred_list.get(img_name, np.empty((0, 5)))

                gt_boxes = gt_bbx_list[j][0].astype("float")
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue

                ignore = np.zeros(gt_boxes.shape[0], dtype=np.int32)
                if len(keep_index) != 0:
                    ignore[keep_index - 1] = 1

                pred_recall, proposal_list = image_eval(
                    pred_info, gt_boxes, ignore, iou_th, pool
                )

                _img_pr_info, fp = img_pr_info(
                    thresh_num, pred_info, proposal_list, pred_recall
                )

                pr_curve += _img_pr_info

        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)
        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        # Print recall-precision at different thresholds
        for srecall in np.arange(0.1, 1.0001, 0.1):
            rindex = len(np.where(recall <= srecall)[0]) - 1
            if rindex >= 0 and rindex < len(recall):
                rthresh = 1.0 - float(rindex) / thresh_num
                print(
                    f"Recall-Precision-Thresh: {recall[rindex]:.4f} "
                    f"{propose[rindex]:.4f} {rthresh:.4f}"
                )

        ap = voc_ap(recall, propose)
        aps[setting_id] = ap
        tb = datetime.datetime.now()
        print(
            f"{settings[setting_id]} cost {(tb - ta).total_seconds():.4f} seconds, "
            f"ap: {ap:.5f}"
        )

    pool.close()
    pool.join()
    return aps


# =============================================================================
# Model Inference Pipeline
# =============================================================================


def load_model(checkpoint_path: str, device: torch.device) -> YuNet:
    """
    Load YuNet model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Target device

    Returns:
        Loaded model in eval mode
    """
    model = YuNet(num_classes=1, num_keypoints=5)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(
    image: np.ndarray, divisor: int = 32
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int, int, int], tuple[int, int]]:
    """
    Preprocess image for inference using letterbox + divisor padding at original scale.

    Pipeline:
        1. Letterbox to square (max dimension)
        2. Pad to nearest multiple of divisor
        3. No resizing - keeps original scale

    Args:
        image: Input image (H, W, 3) in BGR format
        divisor: Size divisor for padding (default 32)

    Returns:
        Tuple of (preprocessed_tensor, original_size, padding, padded_size)
            - padding: (pad_left, pad_top, pad_right, pad_bottom)
    """
    orig_h, orig_w = image.shape[:2]
    max_size = max(orig_h, orig_w)

    # Step 1: Letterbox to square (center the image)
    pad_left = (max_size - orig_w) // 2
    pad_top = (max_size - orig_h) // 2
    pad_right = max_size - orig_w - pad_left
    pad_bottom = max_size - orig_h - pad_top

    # Step 2: Additional padding to make divisible by divisor
    pad_divisor = (divisor - (max_size % divisor)) % divisor
    pad_right += pad_divisor
    pad_bottom += pad_divisor

    # Apply all padding at once
    image = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    padded_h, padded_w = image.shape[:2]
    padding = (pad_left, pad_top, pad_right, pad_bottom)

    tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    return tensor, (orig_h, orig_w), padding, (padded_h, padded_w)


@torch.no_grad()
def run_inference(
    model: YuNet,
    image_tensor: torch.Tensor,
    device: torch.device,
    score_threshold: float,
    iou_threshold: float,
    padding: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Run inference on a single image.

    Args:
        model: YuNet model
        image_tensor: Preprocessed image tensor (1, C, H, W)
        device: Target device
        score_threshold: Score threshold for filtering
        iou_threshold: IoU threshold for NMS
        padding: (pad_left, pad_top, pad_right, pad_bottom) letterbox padding applied

    Returns:
        Detections with shape (N, 5) in [x, y, w, h, score] format (original image coords)
    """
    image_tensor = image_tensor.to(device)

    # Forward pass
    p8_out, p16_out, p32_out = model(image_tensor)

    # Postprocess predictions
    obj_logits, cls_logits, boxes, kps_logits, priors = postprocess_predictions(
        (p8_out, p16_out, p32_out), (8, 16, 32)
    )

    # Compute confidence scores (objectness * class)
    scores = torch.sigmoid(obj_logits) * torch.sigmoid(cls_logits.squeeze(-1))
    scores = scores.squeeze(0)  # Remove batch dimension
    boxes = boxes.squeeze(0)  # Remove batch dimension

    # Filter by score threshold
    mask = scores > score_threshold
    scores = scores[mask]
    boxes = boxes[mask]

    if len(scores) == 0:
        return np.empty((0, 5), dtype=np.float32)

    # Apply NMS
    keep = torchvision.ops.nms(boxes, scores, iou_threshold)
    boxes = boxes[keep]
    scores = scores[keep]

    # Convert back to original image coordinates (subtract letterbox padding)
    pad_left, pad_top = padding[0], padding[1]

    boxes[:, 0] = boxes[:, 0] - pad_left  # x1
    boxes[:, 1] = boxes[:, 1] - pad_top  # y1
    boxes[:, 2] = boxes[:, 2] - pad_left  # x2
    boxes[:, 3] = boxes[:, 3] - pad_top  # y2

    # Convert from xyxy to xywh format
    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()

    x = boxes_np[:, 0]
    y = boxes_np[:, 1]
    w = boxes_np[:, 2] - boxes_np[:, 0]
    h = boxes_np[:, 3] - boxes_np[:, 1]

    # Combine into [x, y, w, h, score]
    detections = np.stack([x, y, w, h, scores_np], axis=1).astype(np.float32)

    # Sort by score descending
    indices = np.argsort(-detections[:, 4])
    detections = detections[indices]

    return detections


# =============================================================================
# Main Evaluation Loop
# =============================================================================


def get_validation_images(data_dir: str) -> list[tuple[str, str, str]]:
    """
    Get list of validation images organized by event.

    Args:
        data_dir: Base WiderFace data directory

    Returns:
        List of (event_name, image_name, full_path) tuples
    """
    val_images_dir = os.path.join(data_dir, "WIDER_val", "WIDER_val", "images")

    # Alternative path structure
    if not os.path.exists(val_images_dir):
        val_images_dir = os.path.join(data_dir, "WIDER_val", "images")

    if not os.path.exists(val_images_dir):
        raise FileNotFoundError(
            f"Could not find validation images directory. "
            f"Tried: {os.path.join(data_dir, 'WIDER_val', 'WIDER_val', 'images')} "
            f"and {os.path.join(data_dir, 'WIDER_val', 'images')}"
        )

    images = []
    events = sorted(os.listdir(val_images_dir))

    for event in events:
        event_dir = os.path.join(val_images_dir, event)
        if not os.path.isdir(event_dir):
            continue

        for img_file in sorted(os.listdir(event_dir)):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_name = os.path.splitext(img_file)[0]
                full_path = os.path.join(event_dir, img_file)
                images.append((event, img_name, full_path))

    return images


def evaluate(args: argparse.Namespace) -> None:
    """
    Main evaluation function.

    Args:
        args: Command line arguments
    """
    device = torch.device(args.device)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    print(f"Getting validation images from {args.data_dir}")
    images = get_validation_images(args.data_dir)
    print(f"Found {len(images)} validation images")

    # Run inference on all images
    predictions: dict[str, dict[str, np.ndarray]] = {}

    print("Running inference (letterbox + divisor padding, original scale)...")
    for event_name, img_name, img_path in tqdm(images, desc="Processing images"):
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            continue

        # Preprocess (letterbox + divisor padding, no resize)
        image_tensor, orig_size, padding, padded_size = preprocess_image(image)

        # Run inference
        detections = run_inference(
            model,
            image_tensor,
            device,
            args.score_threshold,
            args.iou_threshold,
            padding,
        )

        # Store predictions
        if event_name not in predictions:
            predictions[event_name] = {}
        predictions[event_name][img_name] = detections

    # Run WiderFace evaluation
    print(f"\nRunning WiderFace evaluation with gt_dir={args.gt_dir}")
    aps = wider_evaluation(
        predictions, args.gt_dir, iou_thresh=0.5, num_workers=args.num_workers
    )

    # Print final results
    print("\n" + "=" * 50)
    print("WiderFace Evaluation Results")
    print("=" * 50)
    print(f"Easy   AP: {aps[0]:.5f}")
    print(f"Medium AP: {aps[1]:.5f}")
    print(f"Hard   AP: {aps[2]:.5f}")
    print("=" * 50)

    # Save results if output path specified
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(f"{aps[0]:.6f},{aps[1]:.6f},{aps[2]:.6f}\n")
        print(f"Results saved to {args.output}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WiderFace Evaluation for YuNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/widerface",
        help="Path to WiderFace dataset directory",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default="data/widerface/labelv2/val/gt",
        help="Path to ground truth directory containing .mat files",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.02,
        help="Score threshold for filtering detections",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IoU threshold for NMS",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (cuda:0, cpu, etc.)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker processes for evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results (optional)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
