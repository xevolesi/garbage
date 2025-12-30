import torch
import torch.nn as nn


def eiou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    smooth_point: float = 0.1,
    eps: float = 1e-7,
) -> torch.Tensor:
    r"""Implementation of paper 'Extended-IoU Loss: A Systematic IoU-Related
     Method: Beyond Simplified Regression for Better Localization,

     https://ieeexplore.ieee.org/abstract/document/9429909

    Args:
        pred_boxes: Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target_boxes: Corresponding gt bboxes, shape (n, 4).
        smooth_point: hyperparameter, default is 0.1
        eps: Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    px1, py1, px2, py2 = (
        pred_boxes[:, 0],
        pred_boxes[:, 1],
        pred_boxes[:, 2],
        pred_boxes[:, 3],
    )
    tx1, ty1, tx2, ty2 = (
        target_boxes[:, 0],
        target_boxes[:, 1],
        target_boxes[:, 2],
        target_boxes[:, 3],
    )

    # extent top left
    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)

    # intersection coordinates
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)

    # extra
    xmin = torch.min(ix1, ix2)
    ymin = torch.min(iy1, iy2)
    xmax = torch.max(ix1, ix2)
    ymax = torch.max(iy1, iy2)

    # Intersection
    intersection = (
        (ix2 - ex1) * (iy2 - ey1)
        + (xmin - ex1) * (ymin - ey1)
        - (ix1 - ex1) * (ymax - ey1)
        - (xmax - ex1) * (iy1 - ey1)
    )
    # Union
    union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (ty2 - ty1) - intersection + eps
    # IoU
    ious = 1 - (intersection / union)

    # Smooth-EIoU
    smooth_sign = (ious < smooth_point).detach().float()
    loss = 0.5 * smooth_sign * (ious**2) / smooth_point + (1 - smooth_sign) * (
        ious - 0.5 * smooth_point
    )
    return loss


class EIoULoss(nn.Module):
    def __init__(
        self,
        smooth_point: float = 0.1,
    ) -> None:
        super().__init__()
        self.eps = 1e-6
        self.smooth_point = smooth_point

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eiou = eiou_loss(pred, target, smooth_point=self.smooth_point, eps=self.eps)
        return eiou
