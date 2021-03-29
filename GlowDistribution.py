#Uses Kim Seonghyeon's implementation of GLOW.
#Available at: https://github.com/rosinality/glow-pytorch
#See below, folder renamed from "glow-pytorch" to "glow_pytorch" (for importing into python)
#This package should be searchable from python kernel.

import torch
import torch.nn as nn

from glow_pytorch import model as glow

class _GlowDistribution(nn.Module):
    def __init__(self, n_flow=32, n_block=4, num_conditional=None):
        super(_GlowDistribution, self).__init__()
        self.glow_net = glow.Glow(
            3, n_flow, n_block, affine=False, conv_lu=True, num_conditional=num_conditional)
        self.z_shapes = self.calc_z_shapes(3, 64, n_flow, n_block)

    def log_prob(self, samples, conditional=None):
        samples = samples - 0.5
        log_p, log_dets, _ = self.glow_net(samples + torch.rand_like(samples)/32.0, conditional=conditional )
        log_det = log_dets.mean()
        log = log_p + log_det
        nats_per_dim = log / ( 3 * 64 * 64 )
        return {"log_prob": nats_per_dim}

    def sample(self, conditional=None):
        z_sample = []
        for z in self.z_shapes:
            z_new = torch.randn(1, z[0], z[1], z[2]) * 0.7
            z_sample.append(z_new.to("cuda"))
        return self.glow_net.reverse(z_sample, conditional=conditional) + 0.5

    def calc_z_shapes(self, n_channel, input_size, n_flow, n_block):
        z_shapes = [ (n_channel*2**block, input_size//2**block, input_size//2**block) for block in range(1,n_block) ]
        z_shapes.append( (n_channel*2**(n_block+1), input_size//2**n_block, input_size//2**n_block) )
        return z_shapes


class GlowDistribution(nn.Module):
    def __init__(self):
        super(GlowDistribution, self).__init__()
        self.distribution = _GlowDistribution()

    def log_prob(self, samples):
        return self.distribution.log_prob(samples)

    def sample(self):
        return self.distribution.sample()


class _GlowLayerDistribution(nn.Module):
    def __init__(self, distribution, params):
        super(_GlowLayerDistribution, self).__init__()
        self.distribution = distribution
        self.params = params

    def log_prob(self, samples):
        return self.distribution.log_prob(samples, self.params)

    def sample(self):
        with torch.no_grad():
            return self.distribution.sample(self.params)


class GlowLayer(nn.Module):
    def __init__(self, num_conditional):
        super(GlowLayer, self).__init__()
        self.distribution = _GlowDistribution(num_conditional=num_conditional)

    def forward(self, x):
        return _GlowLayerDistribution(self.distribution, x)
