import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from PyGenBrix import VAEModels as vae_models

# Works moderately well on MNIST after training for 34 epochs with a Bernoulli distribution.
# Achieved validation around 66.8

class ConditionalVAE(nn.Module):
    """
    to build:
    mymodel = vae.ConditionalVAE( vae_models.YZVAEModel(), vae.BernoulliConditionalDistribution(), device )
    or
    mymodel = vae.ConditionalVAE( vae_models.YZVAEModel(), cnn.ParallelCNNConditionalDistribution( [ 1, 28, 28 ], vae.QuantizedConditionalDistribution(), device ), device )
    to train:
    Train.train( mydist, mnist, device, batch_size = 32 )
    """
    def __init__( self, vae_model, p_conditional_distribution, device ):
        super(ConditionalVAE, self).__init__()

        self.p_conditional_distribution = p_conditional_distribution.to( device )
        self.device = device

        self.vae_model = vae_model
        self.encoder = vae_model.encoder( no_conditional_channels = vae_model.latents ).to( device )
        self.conditional = vae_model.encoder().to( device )
        self.decoder = vae_model.decoder( p_conditional_distribution.params_size( vae_model.dims[0] ), no_conditional_channels = vae_model.latents ).to( device )

    def encode(self, x):
        params = self.encoder( x )
        split = torch.reshape( params, ( x.shape[0], self.vae_model.latents, 2 ) )
        return split[...,0], split[...,1]

    def decode(self, z):
        reshape_z = torch.reshape( z, ( z.shape[0], self.vae_model.latents + self.vae_model.latents, 1, 1 ) )
        return self.decoder( reshape_z )

    def log_prob( self, x, conditional ):
        encode_conditional = self.conditional( conditional )[:,:self.vae_model.latents]
        cond_reshape = encode_conditional.reshape( conditional.shape[0], encode_conditional.shape[1], 1, 1 ).repeat( 1, 1, 28, 28 )
        concat_encoder = torch.cat( ( x, cond_reshape ), 1 )
        mu, logvar = self.encode( concat_encoder )
        z = torch.distributions.normal.Normal( mu, torch.exp( 0.5*logvar ) ).rsample()
        concat_decoder = torch.cat( ( z, encode_conditional ), 1 )
        decode_params = self.decode( concat_decoder )
        recons_log_prob = self.p_conditional_distribution.log_prob( x, decode_params )
        kl_divergence = torch.sum(
            torch.distributions.kl.kl_divergence(
                torch.distributions.Normal( loc = mu, scale = torch.exp( 0.5*logvar ) ),
	        torch.distributions.Normal( loc = torch.tensor( 0.0 ).to( self.device ), scale = torch.tensor( 1.0 ).to( self.device ) ) ),
            dim = ( 1 ) )
        total_log_prob = recons_log_prob - kl_divergence
        
        return total_log_prob

    def sample( self, conditional ):
        z = torch.tensor( np.random.normal( size = [ 1, self.vae_model.latents ] ).astype( np.float32 ) ).to( self.device )
        encode_conditional = self.conditional( conditional )[:,:self.vae_model.latents]
        concat_decoder = torch.cat( ( z, encode_conditional ), 1 )
        decode_params = self.decode( concat_decoder )
        return self.p_conditional_distribution.sample( decode_params )
