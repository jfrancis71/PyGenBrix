import torch.nn as nn
import torch
import numpy as np


def generate_pixel_groups( height, width ):#1 means predict on this iteration
    pixel_groups = np.zeros( [ 4, height, width ] ).astype( np.float32 )
    pixel_groups[0,::2,::2] = 1
    pixel_groups[1,1::2,1::2] = 1
    pixel_groups[2,::2,1::2] = 1
    pixel_groups[3,1::2,0::2] = 1
    return pixel_groups

def generate_pixel_channel_groups( dims ):
    pixel_channel_groups = np.zeros( [ 4, dims[0], dims[0], dims[1], dims[2] ]).astype( np.float32 )
    pixel_groups = generate_pixel_groups( dims[1], dims[2] )
    for p in range(4):
        for ch in range(dims[0]):
            pixel_channel_groups[p,ch,ch,:,:] = pixel_groups[p,:,:]
    pixel_channel_groups = pixel_channel_groups.reshape( [ dims[0]*4, dims[0], dims[1], dims[2] ] )
    return pixel_channel_groups

def generate_param_groups( dims, params_size ):
    pixel_groups = generate_pixel_channel_groups( dims )
    struct = np.zeros( [ 4*dims[0], dims[0], params_size, dims[1], dims[2] ] )
    for g in range( 4*dims[0] ):
        for p in range( params_size ):
            struct[g,:,p] = pixel_groups[g]
    params = struct.reshape( [ 4*dims[0], dims[0]*params_size, dims[1], dims[2] ] )
    return params

#1 means you are allowed to see this, 0 means must be blocked
def generate_information_masks( dims ):
    pixel_channel_groups = generate_pixel_channel_groups( dims )
    information_masks = np.array( [ np.sum( pixel_channel_groups[:x], axis=0 ) if x > 0 else np.zeros( [ dims[0], dims[1], dims[2] ] ) for x in range(4*dims[0]) ] )
    return information_masks

def create_parallelcnns( dims, params_size, device ):
    return [ torch.nn.Sequential(
#Note the 1*dims[0] refers to we're saying 1 conditional parameter per channel
        torch.nn.Conv2d( dims[0]+1*dims[0],16,3, padding=1 ).to( device ), nn.Tanh().to( device ),
#        torch.nn.Conv2d( 16, 16, 1).to( device ), nn.Tanh().to( device ),
        torch.nn.Conv2d( 16, dims[0]*params_size, 1, padding=0 ).to( device )
        
) for x in range(4*dims[0]) ]

class ConditionalParallelCNNDistribution( nn.Module ):

    def __init__( self, dims, p_conditional_distribution, device ):
        super(ConditionalParallelCNNDistribution, self).__init__()
        self.parallelcnns = nn.ModuleList( create_parallelcnns( dims, p_conditional_distribution.params_size(), device ) )
        self.pixel_channel_groups = generate_pixel_channel_groups( dims )
        self.param_groups = generate_param_groups( dims, p_conditional_distribution.params_size() )
        self.information_masks = generate_information_masks( dims )
        self.device = device
        self.p_conditional_distribution = p_conditional_distribution
        self.dims = dims
        
    def log_prob( self, samples, conditional_input ):
        output_logits = torch.tensor( np.zeros( [ samples.shape[0], samples.shape[1]*self.p_conditional_distribution.params_size(), samples.shape[2], samples.shape[3] ] ) ).to( self.device )
        for n in range( len( self.parallelcnns ) ):
            masked_input = samples*torch.tensor( self.information_masks[n].astype( np.float32 ) ).to( self.device )
            subnet_input = torch.cat( (
                masked_input,
                conditional_input.expand_as( samples ) ), dim=1 )
            subnet_output_logits = self.parallelcnns[n]( subnet_input )
            subnet_masked = subnet_output_logits * torch.tensor( self.param_groups[n] ).to( self.device )
            output_logits +=  subnet_masked

#        output_logprob = torch.distributions.Bernoulli( logits = output_logits ).log_prob( samples )
        output_logprob = self.p_conditional_distribution.log_prob( samples, output_logits )
#        print( "OUTPUTLOGPROB", output_logprob.shape )

        return output_logprob

    def sample( self, conditional_input ):
        sample = torch.tensor( np.zeros( [ 1, self.dims[0], self.dims[1], self.dims[2] ] ).astype( np.float32 ) ).to( self.device )

        for n in range( len( self.parallelcnns ) ):
            subnet_input = torch.cat(
                ( sample*torch.tensor( self.information_masks[n].astype( np.float32 ) ).to( self.device ),
                conditional_input.expand_as( sample ) ), dim=1 )
            subnet_output_logits = self.parallelcnns[n]( subnet_input )

            sample += self.p_conditional_distribution.sample( subnet_output_logits ) * \
                torch.tensor( self.pixel_channel_groups[n] ).to( self.device )

        return sample

    def params_size( self ):
        return 1
