import torch.nn as nn
import torch.nn.functional as F

#Idea comes from Dreamer v2
#https://arxiv.org/pdf/2010.02193.pdf, page 3
#They used stochastic neuron, Following code is deterministic


class BinaryLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
#First return is non differentiable, second is using stg
        bin_latent_code = (x>0.0)*1.0
        probs = F.sigmoid(x)
        stg_latent_code = bin_latent_code + probs - probs.detach()
        return bin_latent_code, stg_latent_code
