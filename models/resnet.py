import torch.nn as nn
import PyGenBrix.models.residual_block as rb

#Turns a 3x32x32 image into a vector
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.c1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.l0 = nn.Linear(256*4*4, 256)
        self.b1 = rb.ResidualBlock(64)
        self.b2 = rb.ResidualBlock(128)

    def forward(self, x):
        x = self.c1(x)
        x = nn.LeakyReLU(0.02)(x)
        x = self.c2(x)
        x = nn.LeakyReLU(0.02)(x)
        x = self.b1(x)
        x = self.c3(x)
        x = nn.LeakyReLU(0.02)(x)
        x = self.b2(x)
        x = self.c4(x)
        x = nn.LeakyReLU(0.02)(x)
#        print( "DEB", x.shape )
        x = x.view(x.size(0), -1)
        x = self.l0(x)
        return x
