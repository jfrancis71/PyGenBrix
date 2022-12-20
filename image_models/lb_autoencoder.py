import torch
import torch.nn as nn
import PyGenBrix.dist_layers.pixelcnn as cnn
import PyGenBrix.dist_layers.spatial_independent as sp
import PyGenBrix.dist_layers.common_layers as dl
import PyGenBrix.image_models.residual_block as rb
import PyGenBrix.image_models.binary_layer as bl


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.c1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.out = nn.Conv2d(256, 256, kernel_size=4, padding=0, bias=False)
        self.b1 = rb.ResidualBlock(64)
        self.b2 = rb.ResidualBlock(128)
        self.bin = bl.BinaryLayer()

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
        x = self.out(x)
        x = x.view(-1,256)
        x = self.bin(x)
        return x


class LBDistribution(nn.Module):
    def __init__(self):
        super(LBDistribution, self).__init__()
        self.enc = Encoder()
        self.cnn = cnn.PixelCNNLayer([ 3, 32, 32 ], output_distribution_layer=sp.SpatialIndependentDistributionLayer( [3, 32, 32] , dl.IndependentQuantizedLayer( num_buckets = 8), num_params=30 ), num_conditional=256 )

    def log_prob(self, x):
        if x.size()[1:4] != torch.Size([3,32,32]):
            raise RuntimeError("LBAE only supports [:,3,32,32] but given {}".format(x.shape))
        enc = self.enc(x)[1]
        dist = self.cnn(enc)
        l = dist.log_prob(x)
        return l

    def sample_reconstruction(self, inp, temperature=1.0):
        enc = self.enc(inp)[0]
        dist = self.cnn(enc)
        return dist.sample([], temperature)

    def mode(self, inp):
        enc = self.enc(inp)[0]
        dist = self.cnn(enc)
        return dist.mode()

