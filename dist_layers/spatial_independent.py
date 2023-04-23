import torch
import torch.nn as nn

import PyGenBrix.dist_layers.common_layers as dl


class SpatialNN(nn.Module):
    def __init__(self, input_channels, output_channels, num_params=None, net_size=16):
        super(SpatialNN, self).__init__()
        self.net_size = net_size
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
    def __init__(self, event_shape, base_distribution_layer, num_params=None, net_size=16):
        super(_SpatialIndependentDistribution, self).__init__()
        self.n1 = SpatialNN(1, base_distribution_layer.params_size(1), num_params, net_size)
        self.n = nn.ModuleList([ SpatialNN(i+1, base_distribution_layer.params_size(1), num_params, net_size) for i in range(event_shape[0]-1) ])
        self.base_distribution_layer = base_distribution_layer
        self.event_shape = event_shape

    def log_prob(self, samples, params=None):
        if list(samples.shape[1:]) != self.event_shape:
            raise RuntimeError("samples has shape {}, but event_shape is {}".format(samples.shape, self.event_shape))
        init1 = torch.zeros_like(samples)
        prob1 = self.base_distribution_layer(self.n1(init1[:,:1], params)).log_prob(samples[:,:1])
        remain_probs = 0.0
        if self.event_shape[0] > 1:
            remain_probs = torch.stack([ self.base_distribution_layer(self.n[i](samples[:,:i+1], params)).log_prob(samples[:,i+1:i+2])["log_prob"] for i in range(self.event_shape[0]-1) ])
            remain_probs = torch.sum(remain_probs, dim=0)
        return {"log_prob": prob1["log_prob"]+remain_probs}

    def sample(self, sample_shape=[], params=None, temperature=1.0):
        if params is not None:
            batch_size = params.shape[0]
        else:
            batch_size = samples_shape[0]
        init1 = torch.zeros([batch_size] + self.event_shape, device=next(self.parameters()).device)
        l1 = self.n1(init1[:,:1], params)
        if temperature >= 0.01:
            init1[:,0] = self.base_distribution_layer(self.n1(init1[:,:1], params)).sample(temperature)[:,0]
        else:
            init1[:,0] = self.base_distribution_layer(self.n1(init1[:,:1], params)).mode()[:,0]
        for i in range(self.event_shape[0]-1):
            if temperature >= 0.01:
                init1[:,i+1] = self.base_distribution_layer(self.n[i](init1[:,:i+1], params)).sample(temperature)[:,0]
            else:
                init1[:,i+1] = self.base_distribution_layer(self.n[i](init1[:,:i+1], params)).mode()[:,0]
        return init1

    def mode(self, params=None):
        return self.sample(params, temperature=0.0)


class SpatialIndependentDistributionLayer(dl.Layer):
    """SpatialIndependentDistributionLayer defines a forward method which returns a base distribution
       Variables in spatial neighbourhood are assumed independent, but channel dependence is assumed
       and modelled using a neural network"""
    def __init__(self, event_shape, base_distribution_layer, num_params, net_size=16):
        super(SpatialIndependentDistributionLayer, self).__init__(_SpatialIndependentDistribution(event_shape, base_distribution_layer, num_params=num_params, net_size=net_size))
        self.num_params = num_params
        self.event_shape = event_shape

    def params_size(self, channels):
        return self.num_params
