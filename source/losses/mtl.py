import torch
from torch import nn

from .eiou import EIoULoss


class DetectionLoss(nn.Module):
    def __init__(
        self,
        obj_weight: float = 1.0,
        cls_weight: float = 1.0,
        box_weight: float = 1.0,
        kps_weight: float = 1.0
    ) -> None:
        super().__init__()
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.kps_weight = kps_weight
        self.clf_crit = nn.BCEWithLogitsLoss(reduction="none")
        self.kps_crit = nn.SmoothL1Loss(reduction="none")
        self.box_crit = EIoULoss()
    
    def forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        targets: tuple[torch.Tensor, ...],
        foreground_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        input_objs, input_cls, input_boxes, input_kps = inputs
        target_objs, target_cls, target_boxes, target_kps = targets

        # Let N = N_priors_level1 + ... + N_priors_level_m.
        # Then input_objs, target_objs have (B, N) shape.
        obj_loss = self.clf_crit(input_objs, target_objs)
        obj_loss = obj_loss.sum(dim=1) # (B,)
        obj_loss = self.obj_weight * obj_loss.mean()

        cls_loss = self.clf_crit(input_cls, target_cls) # (B, N, 1).
        fg = foreground_mask.unsqueeze(-1)  # (B, N, 1).
        cls_loss = cls_loss * fg # (B, N, 1).
        cls_loss = cls_loss.sum(dim=(1, 2)) / fg.sum(dim=(1, 2)).clamp_min(1) # (B,).
        cls_loss = self.cls_weight * cls_loss.mean()

        input_boxes = input_boxes[foreground_mask] # (K, 4).
        target_boxes = target_boxes[foreground_mask] # (K, 4).
        box_loss = self.box_crit(input_boxes, target_boxes) # (K,).
        box_loss = self.box_weight * box_loss.mean()

        input_kps = input_kps[foreground_mask] # (K, 10)
        target_kps = target_kps[foreground_mask] # (K, 10)
        valid_kps = target_kps.ne(-1.0) # (K, 10)
        kps_loss = self.kps_crit(input_kps, target_kps) * valid_kps
        kps_loss = self.kps_weight * kps_loss.sum() / valid_kps.sum().clamp_min(1)

        total_loss = obj_loss + cls_loss + box_loss + kps_loss
        return {
            "total_loss": total_loss,
            "obj_loss": obj_loss,
            "cls_loss": cls_loss,
            "box_loss": box_loss,
            "kps_loss": kps_loss,
        }
