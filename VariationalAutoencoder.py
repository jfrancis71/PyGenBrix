import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from PyGenBrix import VAEModels as vae_models

### Conditional Distributions
### Must support log_prob, sample, and params_size
class BernoulliConditionalDistribution():
### Promises to compute log probability on a per sample basis
### Requires:
###     the batch number of samples and conditionals must match
###     invoking functions must supply a total channel size for conditionals that is same as output
###     of params_size
    def log_prob( self, samples, conditionals, mask = None ):
        log_probs = torch.distributions.Bernoulli( logits = conditionals ).log_prob( samples )
        if mask is not None:
            log_probs = log_probs * mask
        log_probs_sum = torch.sum( log_probs, dim = ( 1, 2, 3 ) )
        return log_probs_sum

### Requires conditionals is of a batch shape
    def sample( self, conditionals ):
        return torch.distributions.Bernoulli( logits = conditionals ).sample()

    def params_size( self, channels ):
        return 1*channels

class NormalConditionalDistribution():
    def log_prob( self, samples, conditionals, mask = None ):
        reshaped_conditionals = torch.reshape( conditionals, ( conditionals.shape[0], samples.shape[1], 2, conditionals.shape[2], conditionals.shape[3] ) )
        locs = reshaped_conditionals[:,:,0]
        scales = .05 + torch.nn.Softplus()( reshaped_conditionals[:,:,1] )
        log_probs = torch.distributions.Normal( loc = locs, scale = scales ).log_prob( samples )
        if mask is not None:
            log_probs = log_probs * mask
        log_probs_sum = torch.sum( log_probs, dim = ( 1, 2, 3 ) )
        return log_probs_sum

    def sample( self, conditionals ):
        reshaped_conditionals = torch.reshape( conditionals, ( conditionals.shape[0], conditionals.shape[1]//2, 2, conditionals.shape[2], conditionals.shape[3] ) )
        locs = reshaped_conditionals[:,:,0]
        scales = .05 + torch.nn.Softplus()( reshaped_conditionals[:,:,1] )
        return torch.distributions.Normal( loc = locs, scale = scales ).sample()

    def params_size( self, channels ):
        return 2*channels

class QuantizedConditionalDistribution():
    def log_prob( self, samples, conditionals, mask = None ):
        quantized = torch.clamp( (samples*10.0).floor(), 0, 9 )
        reshaped_conditionals = torch.reshape( conditionals, ( conditionals.shape[0], conditionals.shape[1]//10, 10, conditionals.shape[2], conditionals.shape[3] ) )
        reshaped_conditionals = reshaped_conditionals.permute( ( 0, 1, 3, 4, 2 ) )
        log_probs = torch.distributions.Categorical( logits = reshaped_conditionals ).log_prob( quantized )
        if mask is not None:
            log_probs = log_probs * mask
        log_probs_sum = torch.sum( log_probs, dim = ( 1, 2, 3 ) )
        return log_probs_sum

    def sample( self, conditionals ):
        reshaped_conditionals = torch.reshape( conditionals, ( conditionals.shape[0], conditionals.shape[1]//10, 10, conditionals.shape[2], conditionals.shape[3] ) )
        reshaped_conditionals = reshaped_conditionals.permute( ( 0, 1, 3, 4, 2 ) )
        return torch.distributions.Categorical( logits = reshaped_conditionals ).sample()/10.0 + .05

    def params_size( self, channels ):
        return 10*channels

class VAE(nn.Module):
    """
    to build:
    mymodel = vae.VAE( vae_models.YZVAEModel( device ), vae.BernoulliConditionalDistribution(), device )
    or
    mymodel = vae.VAE( vae_models.YZVAEModel( device ), cnn.ParallelCNNConditionalDistribution( [ 1, 28, 28 ], vae.QuantizedConditionalDistribution(), device ), device )
    to train:
    Train.train( mydist, mnist, device, batch_size = 32 )
    """
    def __init__( self, vae_model, p_conditional_distribution, device ):
        super(VAE, self).__init__()

        self.p_conditional_distribution = p_conditional_distribution
        self.device = device

        self.vae_model = vae_model
        self.decoder = vae_model.decoder( p_conditional_distribution.params_size( vae_model.dims[0] ) )

    def encode(self, x):
        params = self.vae_model.encoder( x )
        mu, logvar = torch.reshape( params, ( x.shape[0], self.vae_model.latents, 2 ) )
        return mu, logvar

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        reshape_z = torch.reshape( z, ( z.shape[0], self.vae_model.latents, 1, 1 ) )
        return self.decoder( reshape_z )

    def log_prob( self, cx ):
        mu, logvar = self.encode( cx )
        z = self.sample_z(mu, logvar)
        decode_params = self.decode( z )
        recons_log_prob = self.p_conditional_distribution.log_prob( cx, decode_params )
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = ( 1 ) )
        total_log_prob = recons_log_prob - kl_divergence
        
        return total_log_prob

    def sample( self ):
        decode_params = self.decode( torch.tensor( np.random.normal( size = [ 1, self.vae_model.latents ] ).astype( np.float32 ) ).to( self.device ) )
        return self.p_conditional_distribution.sample( decode_params )
