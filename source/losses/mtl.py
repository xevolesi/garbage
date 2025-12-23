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
        self.obj_crit = nn.BCEWithLogitsLoss(reduction="none")
        self.cls_crit = nn.BCEWithLogitsLoss(reduction="none")
        self.kps_crit = nn.SmoothL1Loss(reduction="mean")
        self.box_crit = EIoULoss()
    
    def forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        targets: tuple[torch.Tensor, ...],
        foreground_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_objs, input_cls, input_boxes, input_kps = inputs
        target_objs, target_cls, target_boxes, target_kps = targets

        obj_loss = self.obj_weight * self.obj_crit(input_objs, target_objs)
        obj_loss = obj_loss.mean()

        if foreground_mask.any():
            cls_loss = self.cls_weight * self.cls_crit(input_cls, target_cls)
            cls_loss = cls_loss[foreground_mask].mean()
        else:
            cls_loss = input_cls.new_tensor(0.0)
        
        if foreground_mask.any():
            input_fg = input_boxes[foreground_mask]
            target_fg = target_boxes[foreground_mask]
            box_loss = self.box_weight * self.box_crit(input_fg, target_fg)
            box_loss = box_loss.mean()
        else:
            box_loss = input_boxes.new_tensor(0.0)

        if foreground_mask.any() and target_kps.numel() > 0:
            pred_kps_fg = input_kps[foreground_mask]
            tgt_kps_fg = target_kps[foreground_mask]
            kps_loss = self.kps_weight * self.kps_crit(pred_kps_fg, tgt_kps_fg)
        else:
            kps_loss = input_kps.new_tensor(0.0)
        return obj_loss + cls_loss + box_loss + kps_loss, (obj_loss.detach(), cls_loss.detach(), box_loss.detach(), kps_loss.detach())
