import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import nms

from source.models.yunet import YuNet

from .postprocessing import postprocess_predictions


@torch.no_grad()
def calculate_map_torchmetrics(
    model: YuNet,
    dataloader: DataLoader,
    device: torch.device,
    conf_thresh: float = 0.02,
    iou_thresh: float = 0.45,
    metric_names: tuple[str, ...] = ("map_50", "map_small", "map_medium", "map_large"),
) -> dict[str, float]:
    model.eval()
    map_calculator = MeanAveragePrecision(backend="faster_coco_eval").to(device)
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        gt_boxes = [box.to(device, non_blocking=True) for box in batch["boxes"]]
        gt_labels = [bl.to(device, non_blocking=True) for bl in batch["box_labels"]]
        p8_out, p16_out, p32_out = model(images)
        obj_preds, cls_preds, box_preds, kp_preds, priors = postprocess_predictions(
            (p8_out, p16_out, p32_out), (8, 16, 32)
        )
        batch_size = images.shape[0]
        conf = (obj_preds.sigmoid() * cls_preds.squeeze(dim=-1).sigmoid()).sqrt()
        prep = []
        for batch_idx in range(batch_size):
            sample_conf = conf[batch_idx]
            sample_boxes = box_preds[batch_idx]

            sample_keep = sample_conf >= conf_thresh
            filt_conf = sample_conf[sample_keep]
            filt_boxes = sample_boxes[sample_keep]

            keep_indices = nms(filt_boxes, filt_conf, iou_thresh)
            filt_conf = filt_conf[keep_indices]
            filt_boxes = filt_boxes[keep_indices]

            prep.append(
                {
                    "boxes": filt_boxes,
                    "scores": filt_conf.view(-1),
                    "labels": torch.zeros(
                        len(filt_boxes), dtype=torch.long, device=device
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
    return {name: value for name, value in metrics.items() if name in metric_names}
