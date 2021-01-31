import torch.nn as nn
import torch

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

#Quantizes real number in interval [0,1] into Q buckets
class IndependentQuantizedDistribution():
    def __init__( self, logits ): #[ B, C, Y, X, Q ]
        self.dist = torch.distributions.Independent( torch.distributions.Categorical( logits = logits ), reinterpreted_batch_ndims = 3 )
        self.num_buckets = logits.shape[4]

    def log_prob( self, samples ):
        quantized_samples = torch.clamp( ( samples*self.num_buckets ).floor(), 0, self.num_buckets-1 )
        log_prob = self.dist.log_prob( quantized_samples )
        return { "log_prob" : log_prob }

    def sample( self ):
        return self.dist.sample()/self.num_buckets + 1.0/(self.num_buckets*2.0)

class IndependentQuantizedRealDistribution():
    def __init__( self, logits ): #[ B, C, Y, X, Q ]
        self.logits = logits

    def base_log_prob( self, samples ):
        logits_length = self.logits.shape[4]
        b1 = self.logits[:,:,:,:,0]
        s = (samples > 0.5).float()
        bit_log_prob = torch.distributions.Independent( torch.distributions.Bernoulli( logits = b1 ),
            reinterpreted_batch_ndims = 3 ).log_prob( s )
        if ( logits_length > 1):
            rem = ( (samples <= 0.5)*samples + (samples > 0.5)*(samples-0.5) ) * 2.0
            next_logits = torch.unsqueeze( ( samples <= 0.5 ), 4 )*self.logits[:,:,:,:,1:((logits_length-1)//2)+1] + \
                torch.unsqueeze( ( samples > 0.5 ), 4 )*self.logits[:,:,:,:,((logits_length-1)//2)+1:]
            bit_log_prob += IndependentQuantizedRealDistribution( next_logits ).base_log_prob( rem )
        return bit_log_prob

    def log_prob( self, samples ):
        return { "log_prob" : self.base_log_prob( samples ) }

    def base_sample( self ):
        logits_length = self.logits.shape[4]
        b1 = self.logits[:,:,:,:,0]
        bit = torch.distributions.Independent( torch.distributions.Bernoulli( logits = b1 ),
            reinterpreted_batch_ndims = 3 ).sample()
        rem = 0.0
        if ( logits_length > 1):
            next_logits = torch.unsqueeze( ( bit < 0.5 ), 4 )*self.logits[:,:,:,:,1:((logits_length-1)//2)+1] + \
                torch.unsqueeze( ( bit >= 0.5 ), 4 )*self.logits[:,:,:,:,((logits_length-1)//2)+1:]
            rem = IndependentQuantizedRealDistribution( next_logits ).base_sample()
        pow = bit * ((logits_length+1)//2)
        tot =  pow + rem
        return tot

    def sample( self ):
        #place our sample in the center of the bucket
        return ( self.base_sample() / ( self.logits.shape[4] + 1 ) ) + 0.5 / ( self.logits.shape[4] + 1 )

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
    def __init__( self, num_buckets = 8 ):
        super( IndependentQuantizedLayer, self ).__init__()
        self.num_buckets = num_buckets

    def forward( self, distribution_params ):
        reshaped_logits = torch.reshape( distribution_params, ( distribution_params.shape[0], distribution_params.shape[1]//self.num_buckets, self.num_buckets, distribution_params.shape[2], distribution_params.shape[3] ) ) # [ B, C, Q, Y, X ]
        reshaped_logits = reshaped_logits.permute( ( 0, 1, 3, 4, 2 ) ) # [ B, C, Y, X, Q ]
        return IndependentQuantizedDistribution( logits = reshaped_logits )

    def params_size( self, channels ):
        return self.num_buckets*channels

class IndependentQuantizedRealLayer( nn.Module ):
    def __init__( self, num_buckets = 8 ):
        super( IndependentQuantizedRealLayer, self ).__init__()
        self.num_buckets = num_buckets - 1#Consider renaming

    def forward( self, distribution_params ):
        reshaped_logits = torch.reshape( distribution_params, ( distribution_params.shape[0], distribution_params.shape[1]//self.num_buckets, self.num_buckets, distribution_params.shape[2], distribution_params.shape[3] ) ) # [ B, C, Q, Y, X ]
        reshaped_logits = reshaped_logits.permute( ( 0, 1, 3, 4, 2 ) ) # [ B, C, Y, X, Q ]
        return IndependentQuantizedRealDistribution( logits = reshaped_logits )

    def params_size( self, channels ):
        return self.num_buckets*channels
