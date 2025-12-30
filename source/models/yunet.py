import torch
from torch import nn

from .backbones import YUNetBackbone
from .heads import SingleLevelYuNetHead
from .necks import TinyFPN


class YuNet(nn.Module):
    def __init__(self, num_classes: int = 1, num_keypoints: int = 5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.backbone = YUNetBackbone()
        self.neck = TinyFPN()
        self.p8_head = SingleLevelYuNetHead(
            64, num_classes=self.num_classes, num_keypoints=self.num_keypoints
        )
        self.p16_head = SingleLevelYuNetHead(
            64, num_classes=self.num_classes, num_keypoints=self.num_keypoints
        )
        self.p32_head = SingleLevelYuNetHead(
            64, num_classes=self.num_classes, num_keypoints=self.num_keypoints
        )

    def forward(
        self, tensor: torch.Tensor
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        feature_maps = self.backbone(tensor)
        p8_tensor, p16_tensor, p32_tensor = self.neck(feature_maps)
        p8_predictions = self.p8_head(p8_tensor)
        p16_predictions = self.p16_head(p16_tensor)
        p32_predictions = self.p32_head(p32_tensor)
        return p8_predictions, p16_predictions, p32_predictions
