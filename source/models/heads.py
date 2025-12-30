import torch
from torch import nn

from .blocks import DepthWiseConvUnit


class SingleLevelYuNetHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        num_keypoints: int = 5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_box_coords = 4
        self.num_keypoints = num_keypoints

        self.unit = DepthWiseConvUnit(self.in_channels, self.in_channels)
        self.cls_out = DepthWiseConvUnit(
            self.in_channels, self.num_classes, use_bn_relu=False
        )
        self.box_out = DepthWiseConvUnit(
            self.in_channels, self.num_box_coords, use_bn_relu=False
        )
        self.kps_out = DepthWiseConvUnit(
            self.in_channels, 2 * self.num_keypoints, use_bn_relu=False
        )
        self.obj_out = DepthWiseConvUnit(
            self.in_channels, 1, use_bn_relu=False
        )

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tensor = self.unit(tensor)
        cls_logits = self.cls_out(tensor)
        box_logits = self.box_out(tensor)
        kps_logits = self.kps_out(tensor)
        obj_logits = self.obj_out(tensor)
        return obj_logits, cls_logits, box_logits, kps_logits
