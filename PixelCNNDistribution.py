#Uses https://github.com/jfrancis71/pixel-cnn-pp
#Above was forked from Lucas Caccia repository, original: https://github.com/pclucas14/pixel-cnn-pp
#See below, folder renamed from "pixel-cnn-pp" to "pixelcnn_pp" (for importing 
#into python)
#This package should be searchable from python kernel.

import torch
import torch.nn as nn

from pixelcnn_pp import model as pixelcnn_model
from pixelcnn_pp import utils as pixelcnn_utils

class PixelCNNDiscreteMixDistribution():
    def __init__(self, params):
        self.params = params

    def log_prob(self, samples):
        return {"log_prob": -pixelcnn_utils.discretized_mix_logistic_loss((samples*2.0)-1.0, self.params)}

    def sample(self):
        return (pixelcnn_utils.sample_from_discretized_mix_logistic(self.params, 10)+1.0)/2.0


class PixelCNNDiscreteMixLayer(nn.Module):

    def forward(self, params):
        return PixelCNNDiscreteMixDistribution(params=params)

    def params_size(self, channels):
        return 30 if channels == 1 else 100

        
class PixelCNNDistribution(nn.Module):
    def __init__(self, event_shape, output_distribution_layer=PixelCNNDiscreteMixLayer()):
        super(PixelCNNDistribution, self).__init__()
        self.pixelcnn_net = pixelcnn_model.PixelCNN(nr_resnet=5, nr_filters=160,
                input_channels=event_shape[0], nr_params=output_distribution_layer.params_size(event_shape[0]))
        self.output_distribution_layer = output_distribution_layer
        self.event_shape = event_shape

    def log_prob(self, samples):
        params = self.pixelcnn_net((samples*2.0)-1.0)
        return self.output_distribution_layer(params).log_prob(samples)

    def sample(self):
        with torch.no_grad():
            sampl = torch.zeros([1]+self.event_shape, device="cuda")
            for y in range(self.event_shape[1]):
                for x in range(self.event_shape[2]):
                    params=self.pixelcnn_net((sampl*2)-1,sample=True)
                    s = self.output_distribution_layer(params).sample()
                    sampl[0,:,y,x] = s[0,:,y,x]
        return sampl


class PixelCNNConditionalDistribution(nn.Module):
    def __init__(self, num_conditional):
        super(PixelCNNConditionalDistribution, self).__init__()
        self.pixelcnn_net = pixelcnn_model.PixelCNN(nr_resnet=5, nr_filters=160,
                input_channels=3, nr_logistic_mix=100, nr_conditional=num_conditional)

    def log_prob(self, samples, conditionals):
        distribution_params = self.pixelcnn_net((samples*2.0)-1.0, conditional=conditionals)
        return {"log_prob": -pixelcnn_utils.discretized_mix_logistic_loss(samples, distribution_params)}

    def sample(self, conditionals):
        with torch.no_grad():
            sampl = torch.zeros( (1,3,32,32), device="cuda")
            for y in range(32):
                for x in range(32):
                    distribution_params=self.pixelcnn_net((sampl*2)-1,sample=True, conditional=conditionals)
                    s = pixelcnn_utils.sample_from_discretized_mix_logistic(distribution_params, 100)
                    sampl[0,:,y,x] = s[0,:,y,x]
        return (sampl+1.0)/2.0
