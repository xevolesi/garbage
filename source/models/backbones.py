import torch
from torch import nn

from .blocks import ConvHead, DWBlock
from .utils import initialize_weights


class YUNetBackbone(nn.Module):
    """
    YuNet backbone matching the authors' libfacedetection implementation.

    Architecture follows: downsample_idx=[0, 2, 3, 4], out_idx=[3, 4, 5]
    - model0 (Stem) → MaxPool
    - model1 (block)
    - model2 (block) → MaxPool
    - model3 (block) → OUTPUT p8 → MaxPool
    - model4 (block) → OUTPUT p16 → MaxPool
    - model5 (block) → OUTPUT p32
    """

    def __init__(
        self,
        channels: tuple[tuple[int, int, int] | tuple[int, int], ...] = (
            (3, 16, 16),
            (16, 64),
            (64, 64),
            (64, 64),
            (64, 64),
            (64, 64),
        ),
    ) -> None:
        super().__init__()
        self.channels = channels

        # model0 - Stem (stride 2 conv + depthwise unit)
        self.model0 = ConvHead(*self.channels[0])
        # model1, model2 - early feature extraction
        self.model1 = DWBlock(*self.channels[1])
        self.model2 = DWBlock(*self.channels[2])
        # model3, model4, model5 - produce p8, p16, p32 outputs
        self.model3 = DWBlock(*self.channels[3])
        self.model4 = DWBlock(*self.channels[4])
        self.model5 = DWBlock(*self.channels[5])

        # Pooling layers for downsampling (matching downsample_idx=[0, 2, 3, 4])
        self.pool0 = nn.MaxPool2d(2, 2)  # after model0
        self.pool2 = nn.MaxPool2d(2, 2)  # after model2
        self.pool3 = nn.MaxPool2d(2, 2)  # after model3
        self.pool4 = nn.MaxPool2d(2, 2)  # after model4

        initialize_weights(self)

    def forward(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # model0 → pool (downsample_idx=0)
        tensor = self.model0(tensor)
        tensor = self.pool0(tensor)

        # model1 (no pool)
        tensor = self.model1(tensor)

        # model2 → pool (downsample_idx=2)
        tensor = self.model2(tensor)
        tensor = self.pool2(tensor)

        # model3 → output p8 → pool (downsample_idx=3)
        p8_tensor = self.model3(tensor)
        tensor = self.pool3(p8_tensor)

        # model4 → output p16 → pool (downsample_idx=4)
        p16_tensor = self.model4(tensor)
        tensor = self.pool4(p16_tensor)

        # model5 → output p32 (no pool after)
        p32_tensor = self.model5(tensor)

        return p8_tensor, p16_tensor, p32_tensor
