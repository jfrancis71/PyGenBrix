import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from PyGenBrix import VAEModels as vae_models

class IndependentNormalDistribution():
    def __init__( self, loc, scale ):
        self.dist = torch.distributions.Independent( torch.distributions.Normal( loc = loc, scale = scale ), reinterpreted_batch_ndims = 3 )
    def log_prob( self, samples ):
        return { "log_prob" : self.dist.log_prob( samples ) }

    def sample( self ):
        return self.dist.sample()

class IndependentL2Distribution():
    def __init__( self, loc ):
        self.dist = torch.distributions.Independent( torch.distributions.Normal( loc = loc, scale = torch.ones( loc.shape ) ), reinterpreted_batch_ndims = 3 )

    def log_prob( self, samples ):
        return { "log_prob" : self.dist.log_prob( samples ) }

    def sample( self ):
#Note, not really sampling from prob distribution, but this is common in VAE literature,
#where they return the mean as opposed to sampling from the normal distribution with variance 1.
        return self.loc

class IndependentBernoulliDistribution():
    def __init__( self, logits ):
        self.dist = torch.distributions.Independent( torch.distributions.Bernoulli( logits = logits ), reinterpreted_batch_ndims = 3 )
    def log_prob( self, samples ):
        return { "log_prob" : self.dist.log_prob( samples ) }

    def sample( self ):
        return self.dist.sample()

#Quantizes real number in interval [0,1] into 10 buckets
class QuantizedDistribution():
    def __init__( self, logits ):
        reshaped_logits = torch.reshape( logits, ( logits.shape[0], logits.shape[1]//10, 10, logits.shape[2], logits.shape[3] ) )
        reshaped_logits = reshaped_logits.permute( ( 0, 1, 3, 4, 2 ) )
        self.dist = torch.distributions.Categorical( logits = reshaped_logits )

    def log_prob( self, samples ):
        quantized_samples = torch.clamp( (samples*10.0).floor(), 0, 9 )
        log_prob = self.dist.log_prob( quantized_samples )
        log_prob_sum = torch.sum( log_prob, dim = ( 1, 2, 3 ) )
        return { "log_prob" : log_prob_sum }

    def sample( self ):
        return self.dist.sample()/10.0 + .05

class IndependentL2Layer( nn.Module ):

    def forward( self, distribution_params ):
        return IndependentL2Distribution( loc = distribution_params )

    def params_size( self, channels ):
        return 1*channels

class IndependentBernoulliLayer( nn.Module ):

    def forward( self, distribution_params ):
        return IndependentBernoulliDistribution( logits = distribution_params )

    def params_size( self, channels ):
        return 1*channels

class IndependentNormalLayer( nn.Module ):

    def forward( self, distribution_params ):
        if ( distribution_params.shape[1] % 2 != 0 ):
            raise TypeError("channel size of logits must be an even number to encode means and scale, but it is of size {}"
                            .format( distribution_params.shape[1] ) )
        output_channels = distribution_params.shape[1] // 2
        loc = distribution_params[:,:output_channels]
        scale = .05 + torch.nn.Softplus()( distribution_params[:,output_channels:] )
        return IndependentNormalDistribution( loc = loc, scale = scale )

    def params_size( self, channels ):
        return 2*channels

class QuantizedLayer( nn.Module ):

    def forward( self, distribution_params ):
        return QuantizedDistribution( logits = distribution_params )

    def params_size( self, channels ):
        return 10*channels


class VAE(nn.Module):
    """
    to build:
    mymodel = vae.VAE( vae_models.MNISTVAEModel(), vae.IndependentBernoulliLayer() )
    or
    mymodel = vae.VAE( vae_models.MNISTVAEModel(), cnn.MultiStageParallelCNNLayer( [ 1, 28, 28 ], vae.QuantizedLayer() ) )
    to train:
    Train.train( mydist, mnist, device, batch_size = 32 )
    """
    def __init__( self, vae_model, output_distribution_layer ):
        super(VAE, self).__init__()
        self.output_distribution_layer = output_distribution_layer
        self.vae_model = vae_model
        self.encoder = vae_model.encoder()
        self.decoder = vae_model.decoder( output_distribution_layer.params_size( vae_model.dims[0] ) )

    def encode(self, x):
        params = self.encoder( x )
        split = torch.reshape( params, ( x.shape[0], self.vae_model.latents, 2 ) )
        return split[...,0], split[...,1]

    def decode(self, z):
        reshape_z = torch.reshape( z, ( z.shape[0], self.vae_model.latents, 1, 1 ) )
        return self.decoder( reshape_z )

    def log_prob( self, cx ):
        mu, logvar = self.encode( cx )
        z = torch.distributions.normal.Normal( mu, torch.exp( 0.5*logvar ) ).rsample()
        decode_params = self.decode( z )
        recons_log_prob_dict = self.output_distribution_layer( decode_params ).log_prob( cx )
        device = cx.device
        kl_divergence = torch.sum(
            torch.distributions.kl.kl_divergence(
                torch.distributions.Normal( loc = mu, scale = torch.exp( 0.5*logvar ) ),
	        torch.distributions.Normal( loc = torch.tensor( 0.0 ).to( device ), scale = torch.tensor( 1.0 ).to( device ) ) ),
            dim = ( 1 ) )
        total_log_prob = recons_log_prob_dict["log_prob"] - kl_divergence

        result_dict = { "log_prob": total_log_prob, "kl" : kl_divergence, "recon_log_prob" : recons_log_prob_dict["log_prob"] }
        
        return result_dict

    def sample( self, z = None ):
        device = next(self.decoder.parameters()).device
        if z is not None:
            sample_z = z
        else:
            sample_z = np.random.normal( size = [ 1, self.vae_model.latents ] ).astype( np.float32 )
        decode_params = self.decode( torch.tensor( sample_z ).to( device ) )
        return self.output_distribution_layer( decode_params ).sample()
