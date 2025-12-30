import torch
from torch import nn


def point_wise_conv(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)


def depthwise_conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, padding=1, groups=out_channels
    )


class DepthWiseConvUnit(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, use_bn_relu: bool = True
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.use_bn_relu = use_bn_relu
        self.out_channels = out_channels
        self.pconv = point_wise_conv(self.in_channels, self.out_channels)
        self.dconv = depthwise_conv3x3(self.out_channels, self.out_channels)
        self.bn = nn.Identity()
        self.relu = nn.Identity()
        if self.use_bn_relu:
            self.bn = nn.BatchNorm2d(self.out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.dconv(self.pconv(tensor))))


class DepthWiseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unit1 = DepthWiseConvUnit(self.in_channels, self.in_channels)
        self.unit2 = DepthWiseConvUnit(self.in_channels, self.out_channels)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.unit2(self.unit1(tensor))


class Stem(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.sconv = nn.Conv2d(
            self.in_channels, self.mid_channels, kernel_size=3, stride=2, padding=1
        )
        self.bn = nn.BatchNorm2d(self.mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.unit = DepthWiseConvUnit(self.mid_channels, self.out_channels)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.unit(self.relu(self.bn(self.sconv(tensor))))
