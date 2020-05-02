import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class BernoulliConditionalDistribution():
    def log_prob( self, samples, conditionals ):
        log_probs = torch.distributions.Bernoulli( logits = conditionals ).log_prob( samples )
        log_probs_sum = torch.sum( log_probs, dim = ( 1, 2, 3 ) )
        return log_probs_sum

class VAE(nn.Module):
    """
    to build:
    mymodel = vae.VAE( vae.BernoulliConditionalDistribution(), device )
    or
    mymodel = vae.VAE( cnn.ConditionalParallelCNNDistribution( [ 1, 28, 28 ], device ), device )
    to train:
    Train.train( mydist, mnist, device, batch_size = 32 )
    """
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
        decode_params_reshape = torch.reshape( decode_params, ( decode_params.shape[0], 1, 28, 28 ) )
        recons_log_prob = self.p_conditional_distribution.log_prob( cx, decode_params_reshape )
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = ( 1 ) )
        total_log_prob = recons_log_prob - kl_divergence
        
        return torch.mean( total_log_prob )

    def sample( self ):
        recon_x = self.decode( torch.tensor( np.random.normal( size = [ 1, 20 ] ).astype( np.float32 ) ).to( self.device ) )
        return torch.reshape( recon_x, ( 1, 1, 28, 28 ) )
