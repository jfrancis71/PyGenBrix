import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from PyGenBrix import VAEModels as vae_models

class BernoulliConditionalDistribution():
    def log_prob( self, samples, conditionals ):
        log_probs = torch.distributions.Bernoulli( logits = conditionals ).log_prob( samples )
        log_probs_sum = torch.sum( log_probs, dim = ( 1, 2, 3 ) )
        return log_probs_sum

    def sample( self, conditionals ):
        log_probs = torch.distributions.Bernoulli( logits = conditionals ).log_prob( samples )
        return torch.distributions.Bernoulli( logits = conditionals ).sample()

    def params_size( self ):
        return 1

class NormalConditionalDistribution():
    def log_prob( self, samples, conditionals ):
        reshaped_conditionals = torch.reshape( conditionals, ( conditionals.shape[0], 1, 2, 28, 28 ) )
        locs = reshaped_conditionals[:,:,0]
        scales = .05 + torch.nn.Softplus()( reshaped_conditionals[:,:,1] )
        log_probs = torch.distributions.Normal( loc = locs, scale = scales ).log_prob( samples )
        log_probs_sum = torch.sum( log_probs, dim = ( 1, 2, 3 ) )
        return log_probs_sum

    def sample( self, conditionals ):
        reshaped_conditionals = torch.reshape( conditionals, ( conditionals.shape[0], 1, 2, 28, 28 ) )
        locs = reshaped_conditionals[:,:,0]
        scales = .05 + torch.nn.Softplus()( reshaped_conditionals[:,:,1] )
        return torch.distributions.Normal( loc = locs, scale = scales ).sample()

    def params_size( self ):
        return 2

class QuantizedContinuousConditionalDistribution():
    def log_prob( self, samples, conditionals ):
        quantized = (samples*9.0).round()
        reshaped_conditionals = torch.reshape( conditionals, ( conditionals.shape[0], 1, 10, 28, 28 ) )
        reshaped_conditionals = reshaped_conditionals.permute( ( 0, 1, 3, 4, 2 ) )
        log_probs = torch.distributions.Categorical( logits = reshaped_conditionals ).log_prob( quantized )
        log_probs_sum = torch.sum( log_probs, dim = ( 1, 2, 3 ) )
        return log_probs_sum

    def sample( self, conditionals ):
        reshaped_conditionals = torch.reshape( conditionals, ( conditionals.shape[0], 1, 10, 28, 28 ) )
        reshaped_conditionals = reshaped_conditionals.permute( ( 0, 1, 3, 4, 2 ) )
        return torch.distributions.Categorical( logits = reshaped_conditionals ).sample()/10.0

    def params_size( self ):
        return 10

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
        self.fc4 = nn.Linear(400, 784*p_conditional_distribution.params_size()).to( device )

        self.p_conditional_distribution = p_conditional_distribution
        self.device = device

        self.vae_model = vae_models.MNISTVAEModel( device )

    def encode(self, x):
#        h1 = F.relu(self.fc1(x))
#        return self.fc21(h1), self.fc22(h1)
#        return self.vae_model.encoder( x )
        params = self.vae_model.encoder( x )
        split = torch.reshape( params, ( x.shape[0], self.vae_model.latents, 2 ) )
        return split[...,0], split[...,1]

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
#        h3 = F.relu(self.fc3(z))
#        h4 = self.fc4(h3)
        reshape_z = torch.reshape( z, ( z.shape[0], self.vae_model.latents, 1, 1 ) )
        output = self.vae_model.decoder( reshape_z )
        decode_params_reshape = torch.reshape( output, ( output.shape[0], 1*self.p_conditional_distribution.params_size(), 28, 28 ) )
        return decode_params_reshape

    def log_prob( self, cx ):
        x = torch.reshape( cx, ( cx.shape[0], 784 ) )
        mu, logvar = self.encode( cx )
        z = self.sample_z(mu, logvar)
        decode_params = self.decode( z )
        recons_log_prob = self.p_conditional_distribution.log_prob( cx, decode_params )
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = ( 1 ) )
        total_log_prob = recons_log_prob - kl_divergence
        
        return torch.mean( total_log_prob )

    def sample( self ):
        decode_params = self.decode( torch.tensor( np.random.normal( size = [ 1, self.vae_model.latents ] ).astype( np.float32 ) ).to( self.device ) )
        return self.p_conditional_distribution.sample( decode_params )
