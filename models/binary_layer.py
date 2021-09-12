import torch.nn as nn
import torch.nn.functional as F


class BinaryLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
#First return is non differentiable, second is using stg
        bin_latent_code = (x>0.0)*1.0
        probs = F.sigmoid(x)
        stg_latent_code = bin_latent_code + probs - probs.detach()
        return bin_latent_code, stg_latent_code
