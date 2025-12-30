from torch import nn


def initialize_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.02)
            else:
                m.weight.data.normal_(0, 0.01)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
