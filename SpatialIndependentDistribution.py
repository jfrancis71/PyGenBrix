import torch
import torch.nn as nn

from PyGenBrix import DistributionLayers as dl


class SpatialNN(nn.Module):
    def __init__(self, input_channels, output_channels, num_params=None):
        super(SpatialNN, self).__init__()
        self.net_size = 16
        self.c1 = nn.Conv2d(input_channels, self.net_size, kernel_size=1)
        self.c2 = nn.Conv2d(self.net_size, self.net_size, kernel_size=1)
        self.c3 = nn.Conv2d(self.net_size, output_channels, kernel_size=1)
        if num_params is not None:
            self.p1 = nn.Conv2d(num_params, self.net_size, kernel_size=1)
            self.p2 = nn.Conv2d(num_params, self.net_size, kernel_size=1)
            self.p3 = nn.Conv2d(num_params, output_channels, kernel_size=1)

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
    def __init__(self, event_shape, base_distribution_layer, num_params=None):
        super(_SpatialIndependentDistribution, self).__init__()
        self.n1 = SpatialNN(1, base_distribution_layer.params_size(1), num_params)
        self.n = nn.ModuleList([ SpatialNN(i+1, base_distribution_layer.params_size(1), num_params) for i in range(event_shape[0]-1) ])
        self.base_distribution_layer = base_distribution_layer
        self.event_shape = event_shape

    def log_prob(self, samples, params=None):
        init1 = torch.zeros_like(samples)
        prob1 = self.base_distribution_layer(self.n1(init1[:,:1], params)).log_prob(samples[:,:1])
        probs = torch.stack([ self.base_distribution_layer(self.n[i](samples[:,:i+1], params)).log_prob(samples[:,i+1:i+2])["log_prob"] for i in range(self.event_shape[0]-1) ])
        return {"log_prob": prob1["log_prob"]+torch.sum(probs, dim=0)}

    def sample(self, params=None):
        init1 = torch.zeros([1] + self.event_shape, device="cuda")
        init1[0,0] = self.base_distribution_layer(self.n1(init1[:,:1], params)).sample()
        for i in range(self.event_shape[0]-1):
            init1[0,i+1] = self.base_distribution_layer(self.n[i](init1[:,:i+1], params)).sample()
        return init1

    def mode(self, params=None):
        init1 = torch.zeros([1] + self.event_shape, device="cuda")
        init1[0,0] = self.base_distribution_layer(self.n1(init1[:,:1], params)).mode()
        for i in range(self.event_shape[0]-1):
            init1[0,i+1] = self.base_distribution_layer(self.n[i](init1[:,:i+1], params)).mode()
        return init1


class SpatialIndependentDistributionLayer(dl.Layer):
    def __init__(self, event_shape, base_distribution_layer, num_params):
        super(SpatialIndependentDistributionLayer, self).__init__(_SpatialIndependentDistribution(event_shape, base_distribution_layer, num_params=num_params))
        self.num_params = num_params

    def params_size(self, channels):
        return self.num_params
