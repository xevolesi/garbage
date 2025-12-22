import torch
from torch import nn

from .blocks import DepthWiseBlock, Stem


class YUNetBackbone(nn.Module):
    def __init__(
        self,
        channels: tuple[tuple[int, int, int] | tuple[int, int], ...] = (
            (3, 16, 16), (16, 64), (64, 64), (64, 64), (64, 64), (64, 64)
        ),
    ) -> None:
        super().__init__()
        self.channels = channels

        self.stage0 = Stem(*self.channels[0])
        self.stage1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DepthWiseBlock(*self.channels[1]),
            DepthWiseBlock(*self.channels[2]),
        )
        self.stage2 = nn.Sequential(
            DepthWiseBlock(*self.channels[3]),
            nn.MaxPool2d(2, 2),
        )
        self.stage3 = nn.Sequential(
            DepthWiseBlock(*self.channels[4]),
            nn.MaxPool2d(2, 2),
        )
        self.stage4 = nn.Sequential(
            DepthWiseBlock(*self.channels[5]),
            nn.MaxPool2d(2, 2),    
        )
        self.init_weights()

    def forward(
        self,
        tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tensor = self.stage0(tensor)
        tensor = self.stage1(tensor)
        p8_tensor = self.stage2(tensor)
        p16_tensor = self.stage3(p8_tensor)
        p32_tensor = self.stage4(p16_tensor)
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
