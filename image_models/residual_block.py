import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.c1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.c2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        y = self.c1(x)
        y = nn.LeakyReLU(0.02)(y)
        y = self.c2(y)
        x = x + y
        x = nn.LeakyReLU(0.02)(x)
        return x
