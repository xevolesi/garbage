
import torch

from torchvision.ops import nms
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from .postprocessing import postprocess_predictions
from source.models.yunet import YuNet


def decode_keypoints(
    kps_preds: torch.Tensor, grids: torch.Tensor
) -> torch.Tensor:
    num_points = kps_preds.shape[-1] // 2
    decoded = []
    for i in range(num_points):
        kp_encoded = kps_preds[:, [2 * i, 2 * i + 1]]
        kp_decoded = kp_encoded * grids[:, 2:] + grids[:, :2]
        decoded.append(kp_decoded)
    return torch.cat(decoded, dim=1)


@torch.no_grad()
def calculate_map_torchmetrics(
    model: YuNet,
    dataloader: DataLoader,
    device: torch.device,
    conf_thresh: float = 0.02,
    iou_thresh: float = 0.45,
    metric_names: tuple[str, ...] = (
        "map_50", "map_small", "map_medium", "map_large"
    ),
) -> dict[str, float]:
    model.eval()
    map_calculator = MeanAveragePrecision(backend="faster_coco_eval").to(device)
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        gt_boxes = [
            box.to(device, non_blocking=True) for box in batch["boxes"]
        ]
        gt_kps = [
            kp_set.to(device, non_blocking=True)
            for kp_set in batch["key_points"]
        ]
        gt_labels = [
            bl.to(device, non_blocking=True) for bl in batch["box_labels"]
        ]
        p8_out, p16_out, p32_out = model(images)
        obj_preds, cls_preds, box_preds, kp_preds, priors = (
            postprocess_predictions((p8_out, p16_out, p32_out), (8, 16, 32))
        )
        batch_size = images.shape[0]
        conf = (obj_preds.sigmoid() * cls_preds.squeeze(dim=-1).sigmoid()).sqrt()
        prep = []
        for batch_idx in range(batch_size):
            sample_conf = conf[batch_idx]
            sample_boxes = box_preds[batch_idx]
            sample_kps = kp_preds[batch_idx]
            sample_priors = priors[batch_idx]
            decoded_kps = decode_keypoints(sample_kps, sample_priors)

            sample_keep = sample_conf >= conf_thresh
            filt_conf = sample_conf[sample_keep]
            filt_boxes = sample_boxes[sample_keep]
            filt_kps = decoded_kps[sample_keep]

            keep_indices = nms(filt_boxes, filt_conf, iou_thresh)
            filt_conf = filt_conf[keep_indices]
            filt_boxes = filt_boxes[keep_indices]
            filt_kps = filt_kps[keep_indices]

            prep.append(
                {
                    "boxes": filt_boxes,
                    "scores": filt_conf.view(-1),
                    "labels": torch.zeros(
                        len(filt_boxes),
                        dtype=torch.long, device=device
                    ).view(-1),
                }
            )
        targ = [
            {"boxes": gt_boxes[i], "labels": gt_labels[i].view(-1).long()}
            for i in range(batch_size)
        ]
        map_calculator.update(prep, targ)
    metrics = {
        name: tensor.detach().cpu().item()
        for name, tensor in map_calculator.compute().items()
    }
    return {
        name: value for name, value in metrics.items()
        if name in metric_names
    }
