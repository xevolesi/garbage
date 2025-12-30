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

        # ===== OBJECTNESS LOSS =====
        # Loss for all priors
        obj_loss = self.clf_crit(input_objs, target_objs)  # (B, N)
        obj_loss = obj_loss.mean()  # scalar
        obj_loss = self.obj_weight * obj_loss

        # ===== CLASSIFICATION LOSS =====
        # Loss only for foreground priors
        cls_loss = self.clf_crit(input_cls, target_cls)  # (B, N, num_classes)
        fg = foreground_mask.unsqueeze(-1)  # (B, N, 1)
        cls_loss = cls_loss * fg
        cls_loss = cls_loss.sum() / fg.sum().clamp_min(1)  # scalar
        cls_loss = self.cls_weight * cls_loss

        # ===== BOUNDING BOX LOSS =====
        # Loss only for foreground priors
        input_boxes_fg = input_boxes[foreground_mask]  # (K, 4)
        target_boxes_fg = target_boxes[foreground_mask]  # (K, 4)

        if input_boxes_fg.numel() > 0:
            box_loss = self.box_crit(input_boxes_fg, target_boxes_fg)  # (K,)
            box_loss = self.box_weight * box_loss.mean()
        else:
            box_loss = input_boxes.sum() * 0.0  # scalar zero with correct device

        # ===== KEYPOINT LOSS =====
        # Loss only for foreground priors with visibility weighting
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

            # Mark invalid keypoints (value -1.0)
            valid_kps_mask = target_kps_fg.ne(-1.0)  # (K, 10) bool

            # Apply visibility weights and compute final loss
            # kps_weights: (K, 1), kps_loss_raw: (K, 10), valid_kps_mask: (K, 10)
            weighted_kps_loss = kps_loss_raw * valid_kps_mask * kps_weights_fg

            # Average over valid keypoints, weighted by visibility
            num_valid_kps = (valid_kps_mask * kps_weights_fg).sum().clamp_min(1)
            kps_loss = weighted_kps_loss.sum() / num_valid_kps
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

        Args:
            priors: (K, 4) [cx, cy, stride_w, stride_h]
            kps: (K, 2*num_points) in pixel coordinates, with -1 for invalid keypoints

        Returns:
            Encoded keypoints: (K, 2*num_points) in normalized coordinates
        """
        num_points = kps.shape[-1] // 2
        encoded = []

        for i in range(num_points):
            kp_xy = kps[:, [2 * i, 2 * i + 1]]  # (K, 2)

            # Only encode if keypoint is valid
            valid_mask = kp_xy[:, 0].ne(-1.0)  # (K,)

            # Compute encoded coordinates: (kp - prior_center) / prior_stride
            enc_xy = (kp_xy - priors[:, :2]) / priors[:, 2:]  # (K, 2)

            # Set invalid keypoints back to -1 for masking in loss
            enc_xy[~valid_mask] = -1.0

            encoded.append(enc_xy)

        return torch.cat(encoded, dim=1)  # (K, 2*num_points)
