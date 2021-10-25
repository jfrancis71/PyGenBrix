#Uses https://github.com/jfrancis71/pixel-cnn-pp
#Above was forked from Lucas Caccia repository, original: https://github.com/pclucas14/pixel-cnn-pp
#See below, folder renamed from "pixel-cnn-pp" to "pixelcnn_pp" (for importing 
#into python)
#This package should be searchable from python kernel.

import numpy as np
import torch
import torch.nn as nn

import PyGenBrix.dist_layers.common_layers as dl

from pixelcnn_pp import model as pixelcnn_model
from pixelcnn_pp import utils as pixelcnn_utils


class PixelCNNDiscreteMixDistribution():
    def __init__(self, params):
        self.params = params

    def log_prob(self, samples):
        return {"log_prob": -pixelcnn_utils.discretized_mix_logistic_loss((samples*2.0)-1.0, self.params)}

    def sample(self, temp=1.0):
        return (pixelcnn_utils.sample_from_discretized_mix_logistic(self.params, 10)+1.0)/2.0


class PixelCNNDiscreteMixLayer(nn.Module):

    def forward(self, params):
        return PixelCNNDiscreteMixDistribution(params=params)

    def params_size(self, channels):
        return 30 if channels == 1 else 100

        
class _PixelCNNDistribution(nn.Module):
    def __init__(self, event_shape, output_distribution_layer=PixelCNNDiscreteMixLayer(), num_conditional=None, nr_resnet=5):
        super(_PixelCNNDistribution, self).__init__()
        self.pixelcnn_net = pixelcnn_model.PixelCNN(nr_resnet=nr_resnet, nr_filters=160,
                input_channels=event_shape[0], nr_params=output_distribution_layer.params_size(event_shape[0]), nr_conditional=num_conditional)
        self.output_distribution_layer = output_distribution_layer
        self.event_shape = event_shape

    def log_prob(self, samples, conditionals=None):
        if samples.size()[1:4] != torch.Size(self.event_shape):
            raise RuntimeError("sample shape  {}, but event_shape has shape {}"
                            .format(samples.shape[1:4], self.event_shape))
        params = self.pixelcnn_net((samples*2.0)-1.0, conditional=conditionals)
        return {"log_prob": self.output_distribution_layer(params).log_prob(samples)["log_prob"]}

    def sample(self, conditionals=None, temperature=1.0):
        with torch.no_grad():
            sampl = torch.zeros([1]+self.event_shape, device="cuda")
            for y in range(self.event_shape[1]):
                for x in range(self.event_shape[2]):
                    params=self.pixelcnn_net((sampl*2)-1, sample=True, conditional=conditionals)
                    if temperature > .01:
                        s = self.output_distribution_layer(params).sample(temperature)
                    else:
                        s = self.output_distribution_layer(params).mode()
                    sampl[0,:,y,x] = s[0,:,y,x]
        return sampl

    def mode(self, conditionals=None):
        return self.sample(conditionals, temperature=0.0)


class PixelCNNDistribution(dl.Distribution):
    def __init__(self, event_shape, output_distribution_layer=PixelCNNDiscreteMixLayer(), nr_resnet=5):
        super(PixelCNNDistribution, self).__init__()
        self.distribution = _PixelCNNDistribution(event_shape, output_distribution_layer, num_conditional=None, nr_resnet=nr_resnet)


class PixelCNNLayer(dl.Layer):
    def __init__(self, event_shape, num_conditional, output_distribution_layer=PixelCNNDiscreteMixLayer(), nr_resnet=5):
        super(PixelCNNLayer, self).__init__(_PixelCNNDistribution(event_shape, output_distribution_layer, num_conditional, nr_resnet))
