import torch
import torch.nn as nn

from PyGenBrix import DistributionLayers as dl

class SpatialNN(nn.Module):
    def __init__(self, input_channels, num_params=None):
        super(SpatialNN, self).__init__()
        self.c1 = nn.Conv2d(input_channels, 16, kernel_size=1)
        self.c2 = nn.Conv2d(16, 16, kernel_size=1)
        self.c3 = nn.Conv2d(16, 16, kernel_size=1)
        if num_params is not None:
            self.p1 = nn.Conv2d(num_params, 16, kernel_size=1)
            self.p2 = nn.Conv2d(num_params, 16, kernel_size=1)
            self.p3 = nn.Conv2d(num_params, 16, kernel_size=1)

    def forward(self, x, params=None):
        x = self.c1(x)
        if params is not None:
            x += self.p1(params)
        x = nn.ReLU()(x)
        x = self.c2(x)
        if params is not None:
            x += self.p2(params)
        x = nn.ReLU()(x)
        x = self.c3(x)
        if params is not None:
            x += self.p3(params)
        return x

class _SpatialIndependentDistribution(nn.Module):
    def __init__(self, event_shape, num_params=None):
        super(_SpatialIndependentDistribution, self).__init__()
        self.n1 = SpatialNN(1, num_params)
        self.n2 = SpatialNN(1, num_params)
        self.n3 = SpatialNN(2, num_params)
        self.q1 = dl.IndependentQuantizedLayer(num_buckets=16)
        self.q2 = dl.IndependentQuantizedLayer(num_buckets=16)
        self.q3 = dl.IndependentQuantizedLayer(num_buckets=16)
        self.event_shape = event_shape

    def log_prob(self, samples, params=None):
        init1 = torch.zeros_like(samples)
        prob1 = self.q1(self.n1(init1[:,:1], params)).log_prob(samples[:,:1])
        prob2 = self.q1(self.n2(samples[:,:1], params)).log_prob(samples[:,1:2])
        prob3 = self.q1(self.n3(samples[:,:2], params)).log_prob(samples[:,2:3])
        return {"log_prob": prob1["log_prob"]+prob2["log_prob"]+prob3["log_prob"]}

    def sample(self, params=None):
        init1 = torch.zeros([1] + self.event_shape, device="cuda")
        s1 = self.q1(self.n1(init1[:,:1], params)).sample()
        s2 = self.q2(self.n2(s1, params)).sample()
        s3 = self.q3(self.n3(torch.cat([s1,s2], axis=1), params)).sample()
        return torch.cat([s1,s2,s3], axis=1)

class SpatialIndependentDistributionLayer(dl.Layer):
    def __init__(self, event_shape):
        super(SpatialIndependentDistributionLayer, self).__init__(_SpatialIndependentDistribution(event_shape, num_params=16*3))

    def params_size(self, channels):
        return 16*channels


#class SpatialIndependentQuantizedLayer(nn.Module):

#    def __init__(self, num_buckets=8):
#        super(SpatialIndependentQuantizedLayer, self).__init__()
#        self.num_buckets = num_buckets

#    def forward(self, distribution_params):
#        reshaped_logits = torch.reshape(distribution_params, (distribution_params.shape[0], distribut
#ion_params.shape[1]//self.num_buckets, self.num_buckets, distribution_params.shape[2], distribution_p
#arams.shape[3])) # [ B, C, 10, Y, X ]
#        reshaped_logits = reshaped_logits.permute((0, 1, 3, 4, 2)) # [ B, C, Y, X, Q ]
#        return SpatialIndependentQuantizedDistribution(logits=reshaped_logits)

#    def params_size(self, channels):
#        return self.num_buckets*channels

