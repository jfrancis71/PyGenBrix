import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from PyGenBrix import VAEModels as vae_models

class IndependentNormalDistribution():
    def __init__( self, loc, scale ):
        self.dist = torch.distributions.Independent( torch.distributions.Normal( loc = loc, scale = scale ), 3 )
    def log_prob( self, samples ):
        return { "log_prob" : self.dist.log_prob( samples ) }

    def sample( self ):
        return self.dist.sample()

class IndependentL2Distribution():
    def __init__( self, loc ):
        self.loc = torch.tanh( loc )
    def log_prob( self, samples ): #.918... is log (1/sqrt(2 pi) )
        return { "log_prob" : torch.sum( -0.918939 - 0.5 * (samples - self.loc)**2, dim = ( 1, 2, 3 ) ) }

    def sample( self ):
        return self.loc


    
class IndependentBernoulliDistribution():
    def __init__( self, logits ):
        self.dist = torch.distributions.Independent( torch.distributions.Bernoulli( logits = logits ), 3 )
    def log_prob( self, samples ):
        return { "log_prob" : self.dist.log_prob( samples ) }

    def sample( self ):
        return self.dist.sample()


class IndependentL2Layer( nn.Module ):

    def forward( self, logits ):
        return IndependentL2Distribution( logits = logits )

    def params_size( self, channels ):
        return 1*channels

class IndependentBernoulliLayer( nn.Module ):

    def forward( self, logits ):
        return IndependentBernoulliDistribution( logits = logits )

    def params_size( self, channels ):
        return 1*channels

class IndependentNormalLayer( nn.Module ):

    def forward( self, logits ):
        if ( logits.shape[1] % 2 != 0 ):
            raise TypeError("channel size of logits must be an even number to encode means and scale, but it is of size {}"
                            .format( logits.shape[1] ) )
        output_channels = logits.shape[1] // 2
        loc = logits[:,:output_channels]
        scale = .05 + torch.nn.Softplus()( logits[:,output_channels:] )
        return IndependentNormalDistribution( loc = loc, scale = scale )

    def params_size( self, channels ):
        return 2*channels

class QuantizedDistribution():
    def __init__( self, logits ):
        self.logits = logits

    def log_prob( self, samples ):
        quantized = torch.clamp( (samples*10.0).floor(), 0, 9 )
        reshaped_conditionals = torch.reshape( self.logits, ( self.logits.shape[0], self.logits.shape[1]//10, 10, self.logits.shape[2], self.logits.shape[3] ) )
        reshaped_conditionals = reshaped_conditionals.permute( ( 0, 1, 3, 4, 2 ) )
        log_probs = torch.distributions.Categorical( logits = reshaped_conditionals ).log_prob( quantized )
        log_probs_sum = torch.sum( log_probs, dim = ( 1, 2, 3 ) )
        return { "log_prob" : log_probs_sum }

    def sample( self ):
        reshaped_conditionals = torch.reshape( self.logits, ( self.logits.shape[0], self.logits.shape[1]//10, 10, self.logits.shape[2], self.logits.shape[3] ) )
        reshaped_conditionals = reshaped_conditionals.permute( ( 0, 1, 3, 4, 2 ) )
        return torch.distributions.Categorical( logits = reshaped_conditionals ).sample()/10.0 + .05


class QuantizedLayer( nn.Module ):

    def forward( self, logits ):
        return QuantizedDistribution( logits )

    def params_size( self, channels ):
        return 10*channels



class VAE(nn.Module):
    """
    to build:
    mymodel = vae.VAE( vae_models.YZVAEModel(), vae.BernoulliConditionalDistribution() )
    or
    mymodel = vae.VAE( vae_models.YZVAEModel(), cnn.ParallelCNNConditionalDistribution( [ 1, 28, 28 ], vae.QuantizedConditionalDistribution() ) )
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
        kl_divergence = torch.sum(
            torch.distributions.kl.kl_divergence(
                torch.distributions.Normal( loc = mu, scale = torch.exp( 0.5*logvar ) ),
	        torch.distributions.Normal( loc = torch.tensor( 0.0 ).to( next(self.decoder.parameters()).device ), scale = torch.tensor( 1.0 ).to( next(self.decoder.parameters()).device ) ) ),
            dim = ( 1 ) )
        total_log_prob = recons_log_prob_dict["log_prob"] - kl_divergence

        result_dict = { "log_prob": total_log_prob, "kl" : kl_divergence, "recon_log_prob" : recons_log_prob_dict["log_prob"] }
        
        return result_dict

    def sample( self, z = None ):
        if z is not None:
            sample_z = z
        else:
            sample_z = np.random.normal( size = [ 1, self.vae_model.latents ] ).astype( np.float32 )
        decode_params = self.decode( torch.tensor( sample_z ).to( next(self.decoder.parameters()).device ) )
        return self.output_distribution_layer( decode_params ).sample()
