import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DepthWiseConvUnit


class TinyFPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p8unit = DepthWiseConvUnit(64, 64)
        self.p16unit = DepthWiseConvUnit(64, 64)
        self.p32unit = DepthWiseConvUnit(64, 64)
        self.init_weights()
    
    def forward(
        self, feature_maps: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p8_tensor, p16_tensor, p32_tensor = feature_maps
        p32_tensor = self.p32unit(p32_tensor)
        p16_tensor = self.p16unit(p16_tensor + F.interpolate(p32_tensor, scale_factor=2, mode="nearest"))
        p8_tensor = self.p8unit(p8_tensor + F.interpolate(p16_tensor, scale_factor=2, mode="nearest"))
        return p8_tensor, p16_tensor, p32_tensor

    def init_weights(self) -> None:
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
