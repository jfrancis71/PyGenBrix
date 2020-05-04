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

#1 means you are allowed to see this, 0 means must be blocked
def generate_information_masks( dims ):
    pixel_channel_groups = generate_pixel_channel_groups( dims )
    information_masks = np.array( [ np.sum( pixel_channel_groups[:x], axis=0 ) if x > 0 else np.zeros( [ dims[0], dims[1], dims[2] ] ) for x in range(4*dims[0]) ] )
    return information_masks

def create_parallelcnns( dims, params_size, device ):
    return [ torch.nn.Sequential(
#Note we're using 1 here. be careful on the different params_size
#ParallelCNN has a params_size of 1, but the pixel distribution will have different params_size
        torch.nn.Conv2d( dims[0]+1,16,3, padding=1 ).to( device ), nn.Tanh().to( device ),
#        torch.nn.Conv2d( 16, 16, 1).to( device ), nn.Tanh().to( device ),
        torch.nn.Conv2d( 16, params_size, 1, padding=0 ).to( device )
        
) for x in range(4*dims[0]) ]

class ParallelCNNConditionalDistribution( nn.Module ):

    def __init__( self, dims, p_conditional_distribution, device ):
        super(ParallelCNNConditionalDistribution, self).__init__()
        self.parallelcnns = nn.ModuleList( create_parallelcnns( dims, p_conditional_distribution.params_size( dims[0] ), device ) )
        self.pixel_channel_groups = generate_pixel_channel_groups( dims )
        self.information_masks = generate_information_masks( dims )
        self.device = device
        self.p_conditional_distribution = p_conditional_distribution
        self.dims = dims
        
    def log_prob( self, samples, conditional_inputs ):
        output_log_prob = torch.tensor( np.zeros( samples.shape[0] ) ).to( self.device )
        for n in range( len( self.parallelcnns ) ):
            masked_input = samples*torch.tensor( self.information_masks[n].astype( np.float32 ) ).to( self.device )
            subnet_input = torch.cat( (
                masked_input,
                conditional_inputs ), dim=1 )
            subnet_output_logits = self.parallelcnns[n]( subnet_input )
            toutput_log_prob = self.p_conditional_distribution.log_prob( samples, subnet_output_logits, mask = torch.tensor( self.pixel_channel_groups[n] ).to( self.device ) )
            output_log_prob += toutput_log_prob

        return output_log_prob

    def sample( self, conditional_inputs ):
        samples = torch.tensor( np.zeros( [ conditional_inputs.shape[0], self.dims[0], self.dims[1], self.dims[2] ] ).astype( np.float32 ) ).to( self.device )

        for n in range( len( self.parallelcnns ) ):
            subnet_input = torch.cat(
                ( samples*torch.tensor( self.information_masks[n].astype( np.float32 ) ).to( self.device ),
                conditional_inputs ), dim=1 )
            subnet_output_logits = self.parallelcnns[n]( subnet_input )

            samples += self.p_conditional_distribution.sample( subnet_output_logits ) * \
                torch.tensor( self.pixel_channel_groups[n] ).to( self.device )

        return samples

    def params_size( self, channels ):
        return 1
