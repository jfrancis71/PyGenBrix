import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from PyGenBrix import VAEModels as vae_models

# Works moderately well on MNIST after training for 34 epochs with a Bernoulli distribution.
# Achieved validation around 138 ( joint for 1 pair of digits )


class JointVAE(nn.Module):
    """
    to build:
    mymodel = vae.JointVAE( vae_models.YZVAEModel(), vae.BernoulliConditionalDistribution(), device )
    or
    mymodel = vae.JointVAE( vae_models.YZVAEModel(), cnn.ParallelCNNConditionalDistribution( [ 1, 28, 28 ], vae.QuantizedConditionalDistribution(), device ), device )
    to train:
    Train.train( mydist, mnist, device, batch_size = 32 )
    """
    def __init__( self, vae_model, p_conditional_distribution, device ):
        super(JointVAE, self).__init__()

        self.p_conditional_distribution = p_conditional_distribution.to( device )
        self.device = device

        self.vae_model = vae_model
        self.encoder = vae_model.encoder().to( device )
        self.conditional = torch.nn.Sequential(
            torch.nn.Linear( vae_model.latents, vae_model.latents * 8 ), torch.nn.LeakyReLU(),
            torch.nn.Linear( 8 * vae_model.latents, vae_model.latents * 8 ), torch.nn.LeakyReLU(),
            torch.nn.Linear( 8 * vae_model.latents, vae_model.latents * 8 ), torch.nn.LeakyReLU(),
            torch.nn.Linear( 8 * vae_model.latents, vae_model.latents * 2 ) ).to ( device )
        self.decoder = vae_model.decoder( p_conditional_distribution.params_size( vae_model.dims[0] ) ).to( device )

    def encode(self, x):
        params = self.encoder( x )
        split = torch.reshape( params, ( x.shape[0], self.vae_model.latents, 2 ) )
        return split[...,0], split[...,1]

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        reshape_z = torch.reshape( z, ( z.shape[0], self.vae_model.latents, 1, 1 ) )
        return self.decoder( reshape_z )

    def log_prob( self, x ):

        mu1, logvar1 = self.encode( x[:,0] )
        z1 = self.sample_z(mu1, logvar1)
        mu2, logvar2 = self.encode( x[:,1] )
        z2 = self.sample_z(mu2, logvar2)

        decode_params1 = self.decode( z1 )
        recons_log_prob1 = self.p_conditional_distribution.log_prob( x[:,0], decode_params1 )
        decode_params2 = self.decode( z2 )
        recons_log_prob2 = self.p_conditional_distribution.log_prob( x[:,1], decode_params2 )

        kl_divergence1 = torch.sum(
            torch.distributions.kl.kl_divergence(
                torch.distributions.Normal( loc = mu1, scale = torch.exp( 0.5*logvar1 ) ),
	        torch.distributions.Normal( loc = torch.tensor( 0.0 ).to( self.device ), scale = torch.tensor( 1.0 ).to( self.device ) ) ),
            dim = ( 1 ) )

        prior_params = self.conditional( z1 )
        split = torch.reshape( prior_params, ( x.shape[0], self.vae_model.latents, 2 ) )
        prior_mu2, prior_logvar2 = split[...,0], split[...,1]

        kl_divergence2 = torch.sum(
            torch.distributions.kl.kl_divergence(
                torch.distributions.Normal( loc = mu2, scale = torch.exp( 0.5*logvar2 ) ),
	        torch.distributions.Normal( loc = prior_mu2, scale = torch.exp( 0.5 * prior_logvar2 ) ) ),
            dim = ( 1 ) )
        
        total_log_prob = recons_log_prob1 + recons_log_prob2 - kl_divergence1 - kl_divergence2

        return total_log_prob

    def sample( self ):
        z1 = torch.tensor( np.random.normal( size = [ 1, self.vae_model.latents ] ).astype( np.float32 ) ).to( self.device )
        decode_params1 = self.decode( z1 )
        x1 = self.p_conditional_distribution.sample( decode_params1 )
        prior_params = self.conditional( z1 )
        split = torch.reshape( prior_params, ( 1, self.vae_model.latents, 2 ) )
        prior_mu2, prior_logvar2 = split[...,0], split[...,1]
        decode_params2 = self.decode( prior_mu2 )
        x2 = self.p_conditional_distribution.sample( decode_params2 )
        return [ x1, x2 ]
