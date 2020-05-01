import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def log_prob( self, cx ):
        x = torch.reshape( cx, ( cx.shape[0], 784 ) )
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.sample_z(mu, logvar)
        decode_params = self.decode( z )
        recons_log_prob = torch.sum( torch.distributions.Bernoulli( logits = decode_params ).log_prob( x ) )
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recons_log_prob - kl_divergence

    def sample( self ):
        recon_x = self.decode( torch.tensor( np.random.normal( size = [ 1, 20 ] ).astype( np.float32 ) ) )
        return torch.reshape( recon_x, ( 1, 1, 28, 28 ) )
