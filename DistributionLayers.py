import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

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
class IndependentQuantizedDistribution():
    def __init__( self, logits ):
        reshaped_logits = torch.reshape( logits, ( logits.shape[0], logits.shape[1]//10, 10, logits.shape[2], logits.shape[3] ) ) # [ B, C, 10, Y, X ]
        reshaped_logits = reshaped_logits.permute( ( 0, 1, 3, 4, 2 ) ) # [ B, C, Y, X, 10 ]
        self.dist = torch.distributions.Independent( torch.distributions.Categorical( logits = reshaped_logits ), reinterpreted_batch_ndims = 3 )

    def log_prob( self, samples ):
        quantized_samples = torch.clamp( (samples*10.0).floor(), 0, 9 )
        log_prob = self.dist.log_prob( quantized_samples )
        return { "log_prob" : log_prob }

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

class IndependentQuantizedLayer( nn.Module ):

    def forward( self, distribution_params ):
        return IndependentQuantizedDistribution( logits = distribution_params )

    def params_size( self, channels ):
        return 10*channels
