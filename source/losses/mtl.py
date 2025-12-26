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
        self.kps_crit = nn.SmoothL1Loss(reduction="none", beta=0.1111111111111111)
        self.box_crit = EIoULoss()
    
    def forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        targets: tuple[torch.Tensor, ...],
        foreground_mask: torch.Tensor,    
        priors: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        input_objs, input_cls, input_boxes, input_kps = inputs
        target_objs, target_cls, target_boxes, target_kps = targets

        # Let N = N_priors_level1 + ... + N_priors_level_m.
        # Then input_objs, target_objs have (B, N) shape.
        obj_loss = self.clf_crit(input_objs, target_objs)
        obj_loss = obj_loss.mean(dim=1) # (B,)
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

        if foreground_mask.any():
            # priors: (B, N, 4) -> (K, 4) для тех же foreground позиций
            priors_fg = priors[foreground_mask]          # (K, 4)
            input_kps = input_kps[foreground_mask]       # (K, 10)
            target_kps = target_kps[foreground_mask]     # (K, 10) в пикселях

            # Кодируем только валидные таргеты, но проще закодировать всё и маской вырезать
            encoded_target_kps = self._encode_keypoints(priors_fg, target_kps)  # (K, 10)
            kps_loss_raw = self.kps_crit(input_kps, encoded_target_kps)  # (K, 10)
            valid_kps = target_kps.ne(-1.0)

            kps_loss = (kps_loss_raw * valid_kps).sum() / valid_kps.sum().clamp_min(1)
            kps_loss = self.kps_weight * kps_loss
        else:
            kps_loss = input_kps.sum() * 0.0

        total_loss = obj_loss + cls_loss + box_loss + kps_loss
        return {
            "total_loss": total_loss,
            "obj_loss": obj_loss,
            "cls_loss": cls_loss,
            "box_loss": box_loss,
            "kps_loss": kps_loss,
        }

    def _encode_keypoints(self, priors: torch.Tensor, kps: torch.Tensor) -> torch.Tensor:
        # priors: (K, 4) [cx, cy, sx, sy]
        # kps:    (K, 2*num_points) в пикселях
        num_points = kps.shape[-1] // 2
        encoded = []
        for i in range(num_points):
            kp_xy = kps[:, [2*i, 2*i+1]]         # (K, 2)
            enc_xy = (kp_xy - priors[:, :2]) / priors[:, 2:]
            encoded.append(enc_xy)
        return torch.cat(encoded, dim=1)         # (K, 2*num_points)