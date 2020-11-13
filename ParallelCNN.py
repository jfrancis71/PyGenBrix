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

activation_fn = nn.Tanh()

def default_parallel_cnn_fn( dims, params_size ):
    return torch.nn.Sequential(
#Note we're using 1 here. be careful on the different params_size
#ParallelCNN has a params_size of 1, but the pixel distribution will have different params_size
        torch.nn.Conv2d( dims[0]+1,16,3, padding=1 ), activation_fn,
        torch.nn.Conv2d( 16, 32, 3, padding=1), activation_fn,
        torch.nn.Conv2d( 32, 64, 3, padding=1), activation_fn,
        torch.nn.Conv2d( 64, 32, 3, padding=1), activation_fn,
        torch.nn.Conv2d( 32, 16, 1), activation_fn,
        torch.nn.Conv2d( 16, params_size, 1, padding=0 )
)

def create_parallelcnns( dims, params_size, parallel_cnn_fn = default_parallel_cnn_fn ):
    return [ parallel_cnn_fn( dims, params_size ) for x in range(4*dims[0]) ]

class ParallelCNNConditionalDistribution( nn.Module ):

    def __init__( self, dims, p_conditional_distribution ):
        super(ParallelCNNConditionalDistribution, self).__init__()
        self.parallelcnns = nn.ModuleList( create_parallelcnns( dims, p_conditional_distribution.params_size( dims[0] ) ) )
        self.pixel_channel_groups = nn.Parameter( torch.tensor( generate_pixel_channel_groups( dims ).astype( np.float32 ) ), requires_grad = False )
        self.information_masks = nn.Parameter( torch.tensor( generate_information_masks( dims ).astype( np.float32 ) ), requires_grad = False )
        self.p_conditional_distribution = p_conditional_distribution
        self.dims = dims
        
    def log_prob( self, samples, conditional_inputs ):
        output_log_prob = torch.tensor( np.zeros( samples.shape[0] ) ).to( samples.device )
        for n in range( len( self.parallelcnns ) ):
            masked_input = samples*self.information_masks[n]
            subnet_input = torch.cat( (
                masked_input,
                conditional_inputs ), dim=1 )
            subnet_output_logits = self.parallelcnns[n]( subnet_input )
            toutput_log_prob = self.p_conditional_distribution.log_prob( samples, subnet_output_logits, mask = self.pixel_channel_groups[n] )
            output_log_prob += toutput_log_prob

        return output_log_prob

    def sample( self, conditional_inputs ):
        samples = torch.tensor( np.zeros( [ conditional_inputs.shape[0], self.dims[0], self.dims[1], self.dims[2] ] ).astype( np.float32 ) ).to( conditional_inputs.device )

        for n in range( len( self.parallelcnns ) ):
            subnet_input = torch.cat(
                ( samples*self.information_masks[n],
                conditional_inputs ), dim=1 )
            subnet_output_logits = self.parallelcnns[n]( subnet_input )

            samples += self.p_conditional_distribution.sample( subnet_output_logits ) * \
                self.pixel_channel_groups[n]

        return samples

    def params_size( self, channels ):
        return 1

#MultiStageParallelCNN
#This is currently specifically designed for a 64x64 image, but could be made customisable in the future.
#Guide achieved 4,221 at epoch 1,226 for aligned celeba 64x64, with quantized output distribution.

#Performs an "upsample" stage, note the input is not a "smoothed" version of the
#output, it is the even pixels of the output.
#eg in upsampling, if the input is 4x4, these 4x4 will be the even pixels in
#the output 8x8, and the remaining pixels in the 8x8 are sampled conditioned on these.
class Upsampler( nn.Module ):
    #dims is the dimensions of the input sample
    def __init__( self, dims, p_conditional_distribution, parallel_cnn_fn = default_parallel_cnn_fn ):
        super(Upsampler, self).__init__()
        self.output_dims = [ dims[0], dims[1]*2, dims[2]*2 ]
        self.parallelcnns = nn.ModuleList( create_parallelcnns( dims, p_conditional_distribution.params_size( dims[0] ), parallel_cnn_fn ) )
        self.pixel_channel_groups = nn.Parameter( torch.tensor( generate_pixel_channel_groups( self.output_dims ).astype( np.float32 ) ), requires_grad = False )
        self.information_masks = nn.Parameter( torch.tensor( generate_information_masks( self.output_dims ).astype( np.float32 ) ), requires_grad = False )
        self.p_conditional_distribution = p_conditional_distribution
        self.dims = dims

#Compute the log prob of samples conditioned on even pixels (where pixels counts from 0)
#but excluding the log prob of the even pixels themselves
    def log_prob( self, samples, conditional_inputs ):
        assert( samples.shape[0] == conditional_inputs.shape[0] )
        if (samples.shape[2] != conditional_inputs.shape[2]):
            raise TypeError("samples shape  {}, but conditional_inputs has shape {}"
                            .format( samples.shape, conditional_inputs.shape ) )
        
        assert( samples.shape[3] == conditional_inputs.shape[3] )
        output_log_prob = 0.0
        for n in range( self.dims[0], len( self.parallelcnns ) ):
            masked_input = samples*self.information_masks[n]
            subnet_input = torch.cat( (
                masked_input,
                conditional_inputs ), dim=1 )
            subnet_output_logits = self.parallelcnns[n]( subnet_input )
            toutput_log_prob = self.p_conditional_distribution.log_prob( samples, subnet_output_logits, mask = self.pixel_channel_groups[n] )
            output_log_prob += toutput_log_prob

        return output_log_prob

#array is the input array to be upsampled
    def sample( self, array, conditional_inputs ):
            
        assert( array.shape[0] == conditional_inputs.shape[0] )
        if (self.dims[1]*2 != conditional_inputs.shape[2]):
            raise TypeError("dims specified  {}, but conditional_inputs has shape {}"
                            .format( self.dims, conditional_inputs.shape ) )
        
        assert( self.dims[2]*2 == conditional_inputs.shape[3] )
        if ( self.dims != list( array.shape[1:] ) ):
            raise TypeError("dims specified  {}, but array has shape {}"
                            .format( self.dims, array.shape ) )
        
        samples = torch.tensor( np.zeros( [ conditional_inputs.shape[0], self.dims[0], self.output_dims[1], self.output_dims[2] ] ).astype( np.float32 ) ).to( conditional_inputs.device )

        samples[:,:,::2,::2] = array
        
        for n in range( self.dims[0], len( self.parallelcnns ) ):
            subnet_input = torch.cat(
                ( samples*self.information_masks[n],
                conditional_inputs ), dim=1 )
            subnet_output_logits = self.parallelcnns[n]( subnet_input )

            samples += self.p_conditional_distribution.sample( subnet_output_logits ) * \
                self.pixel_channel_groups[n]

        return samples

class MultiStageParallelCNNDistribution( nn.Module ):
    def __init__( self, dims, p_conditional_distribution, parallel_cnn_fn = default_parallel_cnn_fn ):
        super(MultiStageParallelCNNDistribution, self).__init__()
        self.p_conditional_distribution = p_conditional_distribution
        self.parallelcnns = nn.ModuleList( create_parallelcnns( [ dims[0], dims[1]//8, dims[2]//8 ], p_conditional_distribution.params_size( dims[0] ), parallel_cnn_fn ) )
        self.pixel_channel_groups = nn.Parameter( torch.tensor( generate_pixel_channel_groups( [ dims[0], dims[1]//8, dims[2]//8 ] ).astype( np.float32 ) ), requires_grad = False )
        self.information_masks = nn.Parameter( torch.tensor( generate_information_masks( [ dims[0], dims[1]//8, dims[2]//8 ] ).astype( np.float32 ) ), requires_grad = False )

        self.upsampler1 = Upsampler(  [ dims[0], dims[1]//16, dims[2]//16 ], p_conditional_distribution, parallel_cnn_fn )#Largest scale
        self.upsampler2 = Upsampler(  [ dims[0], dims[1]//8, dims[2]//8 ], p_conditional_distribution, parallel_cnn_fn )#Large scale
        self.upsampler3 = Upsampler(  [ dims[0], dims[1]//4, dims[2]//4 ], p_conditional_distribution, parallel_cnn_fn )#Fine scale
        self.upsampler4 = Upsampler(  [ dims[0], dims[1]//2, dims[2]//2 ], p_conditional_distribution, parallel_cnn_fn )#Finest scale
        self.dims = dims

    def log_prob( self, samples, conditional_inputs ):
        subsamples = samples[:,:,::8,::8]
        subconditional_inputs = conditional_inputs[:,:,::8,::8]
        
        output_log_prob = torch.tensor( np.zeros( subsamples.shape[0] ) ).to( subsamples.device )
        for n in range( self.dims[0] ):
            masked_input = subsamples*self.information_masks[n]
            subnet_input = torch.cat( (
                masked_input,
                subconditional_inputs ), dim=1 )
            subnet_output_logits = self.parallelcnns[n]( subnet_input )
            toutput_log_prob = self.p_conditional_distribution.log_prob( subsamples, subnet_output_logits, mask = self.pixel_channel_groups[n] )
            output_log_prob += toutput_log_prob
        
        upsampled_log_prob1 = self.upsampler1.log_prob( samples[:,:,::8,::8], conditional_inputs[:,:,::8,::8] )
        upsampled_log_prob2 = self.upsampler2.log_prob( samples[:,:,::4,::4], conditional_inputs[:,:,::4,::4] )
        upsampled_log_prob3 = self.upsampler3.log_prob( samples[:,:,::2,::2], conditional_inputs[:,:,::2,::2] )
        upsampled_log_prob4 = self.upsampler4.log_prob( samples, conditional_inputs )
        
        return output_log_prob + upsampled_log_prob1 + upsampled_log_prob2 + upsampled_log_prob3 + upsampled_log_prob4

    def sample( self, conditional_inputs ):
        subsamples = torch.tensor( np.zeros( [ conditional_inputs.shape[0], self.dims[0], self.dims[1]//8, self.dims[2]//8 ] ).astype( np.float32 ) ).to( conditional_inputs.device )
        subconditional_inputs = conditional_inputs[:,:,::8,::8]
        
        for n in range( self.dims[0] ):
            subnet_input = torch.cat(
                ( subsamples*self.information_masks[n],
                subconditional_inputs ), dim=1 )
            subnet_output_logits = self.parallelcnns[n]( subnet_input )

            subsamples += self.p_conditional_distribution.sample( subnet_output_logits ) * \
                self.pixel_channel_groups[n]


        u1 = self.upsampler1.sample( subsamples[:,:,::2,::2], subconditional_inputs )
        u2 = self.upsampler2.sample( u1, conditional_inputs[:,:,::4,::4] )
        u3 = self.upsampler3.sample( u2, conditional_inputs[:,:,::2,::2] )
        u4 = self.upsampler4.sample( u3, conditional_inputs )

        return u4
