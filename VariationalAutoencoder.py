import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class BernoulliConditionalDistribution():
    def log_prob( self, samples, conditionals ):
        log_probs = torch.distributions.Bernoulli( logits = conditionals ).log_prob( samples )
        return log_probs

class VAE(nn.Module):
    def __init__( self, p_conditional_distribution, device ):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400).to( device )
        self.fc21 = nn.Linear(400, 20).to( device )
        self.fc22 = nn.Linear(400, 20).to( device )
        self.fc3 = nn.Linear(20, 400).to( device )
        self.fc4 = nn.Linear(400, 784).to( device )

        self.p_conditional_distribution = p_conditional_distribution
        self.device = device

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
        recons_log_prob = self.p_conditional_distribution.log_prob( x, decode_params )
        recons_log_prob_sum = torch.sum( recons_log_prob )
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recons_log_prob_sum - kl_divergence

    def sample( self ):
        recon_x = self.decode( torch.tensor( np.random.normal( size = [ 1, 20 ] ).astype( np.float32 ) ).to( self.device ) )
        return torch.reshape( recon_x, ( 1, 1, 28, 28 ) )
