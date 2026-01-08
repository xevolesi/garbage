import torch
from torch import nn

from .eiou import EIoULoss


class DetectionLoss(nn.Module):
    """YuNet detection loss combining objectness, classification, bbox, and keypoint losses.

    Args:
        obj_weight: Weight for objectness loss. Default: 1.0
        cls_weight: Weight for classification loss. Default: 1.0
        box_weight: Weight for bounding box loss. Default: 1.0
        kps_weight: Weight for keypoint loss. Default: 1.0
    """

    def __init__(
        self,
        obj_weight: float = 1.0,
        cls_weight: float = 1.0,
        box_weight: float = 1.0,
        kps_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.kps_weight = kps_weight

        # Loss functions
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
        """
        Compute detection losses.

        Args:
            inputs: Tuple of (input_objs, input_cls, input_boxes, input_kps)
                - input_objs: (B, N) - objectness logits
                - input_cls: (B, N, num_classes) - classification logits
                - input_boxes: (B, N, 4) - box predictions
                - input_kps: (B, N, 10) - keypoint predictions

            targets: Tuple of (target_objs, target_cls, target_boxes, target_kps, kps_weights)
                - target_objs: (B, N) - objectness targets
                - target_cls: (B, N, num_classes) - classification targets
                - target_boxes: (B, N, 4) - box targets
                - target_kps: (B, N, 10) - keypoint targets
                - kps_weights: (B, N, 1) - visibility weights for keypoints

            foreground_mask: (B, N) bool - foreground prior mask
            priors: (B, N, 4) - priors in [cx, cy, stride_w, stride_h] format

        Returns:
            Dictionary with loss values:
                - total_loss: sum of all losses
                - obj_loss: objectness loss
                - cls_loss: classification loss
                - box_loss: bounding box loss
                - kps_loss: keypoint loss
        """
        input_objs, input_cls, input_boxes, input_kps = inputs
        target_objs, target_cls, target_boxes, target_kps, kps_weights = targets

        # Compute num_total_samples (number of positive/foreground samples)
        # This matches author's: num_total_samples = max(reduce_mean(num_pos), 1.0)
        num_total_samples = foreground_mask.sum().float().clamp_min(1.0)

        # ===== OBJECTNESS LOSS =====
        # Loss for all priors, sum reduction, divided by num_total_samples
        obj_loss = self.clf_crit(input_objs, target_objs)  # (B, N)
        obj_loss = obj_loss.sum() / num_total_samples  # scalar
        obj_loss = self.obj_weight * obj_loss

        # ===== CLASSIFICATION LOSS =====
        # Loss only for foreground priors, sum reduction, divided by num_total_samples
        cls_loss = self.clf_crit(input_cls, target_cls)  # (B, N, num_classes)
        cls_loss = cls_loss[foreground_mask]  # (K, num_classes) - only foreground
        cls_loss = cls_loss.sum() / num_total_samples  # scalar
        cls_loss = self.cls_weight * cls_loss

        # ===== BOUNDING BOX LOSS =====
        # Loss only for foreground priors, sum reduction, divided by num_total_samples
        input_boxes_fg = input_boxes[foreground_mask]  # (K, 4)
        target_boxes_fg = target_boxes[foreground_mask]  # (K, 4)

        if input_boxes_fg.numel() > 0:
            box_loss = self.box_crit(input_boxes_fg, target_boxes_fg)  # (K,)
            box_loss = self.box_weight * box_loss.sum() / num_total_samples
        else:
            box_loss = input_boxes.sum() * 0.0  # scalar zero with correct device

        # ===== KEYPOINT LOSS =====
        # Loss only for foreground priors with visibility weighting
        # Matches author's approach: use only per-face visibility weights,
        # no per-keypoint -1 coordinate masking
        input_kps_fg = input_kps[foreground_mask]  # (K, 10)
        target_kps_fg = target_kps[foreground_mask]  # (K, 10)
        kps_weights_fg = kps_weights[foreground_mask]  # (K, 1)
        priors_fg = priors[foreground_mask]  # (K, 4)

        if input_kps_fg.numel() > 0:
            # Encode keypoint targets
            encoded_target_kps = self._encode_keypoints(
                priors_fg, target_kps_fg
            )  # (K, 10)

            # Compute raw loss
            kps_loss_raw = self.kps_crit(input_kps_fg, encoded_target_kps)  # (K, 10)

            # Apply per-face visibility weights (broadcasts from (K,1) to (K,10))
            # This matches author's: weight=kps_weights.view(-1, 1), avg_factor=sum(kps_weights)
            weighted_kps_loss = kps_loss_raw * kps_weights_fg

            # Sum and divide by total visibility weight
            kps_loss = weighted_kps_loss.sum() / kps_weights_fg.sum().clamp_min(1)
            kps_loss = self.kps_weight * kps_loss
        else:
            kps_loss = input_kps.sum() * 0.0  # scalar zero with correct device

        # ===== TOTAL LOSS =====
        total_loss = obj_loss + cls_loss + box_loss + kps_loss

        return {
            "total_loss": total_loss,
            "obj_loss": obj_loss.detach(),
            "cls_loss": cls_loss.detach(),
            "box_loss": box_loss.detach(),
            "kps_loss": kps_loss.detach(),
        }

    def _encode_keypoints(
        self,
        priors: torch.Tensor,
        kps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode keypoints from pixel coordinates to prior-relative coordinates.

        Matches author's _kps_encode: simply encodes all keypoints without
        checking for invalid values. Invalid keypoints are handled via
        visibility weights in the loss.

        Args:
            priors: (K, 4) [cx, cy, stride_w, stride_h]
            kps: (K, 2*num_points) in pixel coordinates

        Returns:
            Encoded keypoints: (K, 2*num_points) in normalized coordinates
        """
        num_points = kps.shape[-1] // 2
        encoded_kps = [
            (kps[:, [2 * i, 2 * i + 1]] - priors[:, :2]) / priors[:, 2:]
            for i in range(num_points)
        ]
        return torch.cat(encoded_kps, dim=1)
