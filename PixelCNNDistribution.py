#Uses https://github.com/jfrancis71/pixel-cnn-pp
#Above was forked from Lucas Caccia repository, original: https://github.com/pclucas14/pixel-cnn-pp
#See below, folder renamed from "pixel-cnn-pp" to "pixelcnn_pp" (for importing 
#into python)
#This package should be searchable from python kernel.

import torch
import torch.nn as nn

from pixelcnn_pp import model as pixelcnn_model
from pixelcnn_pp import utils as pixelcnn_utils

class PixelCNNDistribution(nn.Module):
    def __init__(self):
        super(PixelCNNDistribution, self).__init__()
        self.pixelcnn_net = pixelcnn_model.PixelCNN(nr_resnet=5, nr_filters=160,
                input_channels=3, nr_logistic_mix=100)

    def log_prob(self, samples):
        distribution_params = self.pixelcnn_net((samples*2.0)-1.0)
        return {"log_prob": -pixelcnn_utils.discretized_mix_logistic_loss(samples, distribution_params)}

    def sample(self):
        with torch.no_grad():
            sampl = torch.zeros( (1,3,32,32), device="cuda")
            for y in range(32):
                for x in range(32):
                    distribution_params=self.pixelcnn_net((sampl*2)-1,sample=True)
                    s = pixelcnn_utils.sample_from_discretized_mix_logistic(distribution_params, 100)
                    sampl[0,:,y,x] = s[0,:,y,x]
        return (sampl+1.0)/2.0
        
