import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


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

def default_parallel_cnn_fn( dims, params_size ):
    activation_fn = nn.Tanh()
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

class MultiStageParallelCNNConditionalDistribution( nn.Module ):
#levels is number of upsampling levels, eg. 4 takes you from a 4x4 to 64x64. It has a minimum value of 1
    def __init__( self, dims, p_conditional_distribution, levels, parallel_cnn_fn = default_parallel_cnn_fn ):
        super(MultiStageParallelCNNConditionalDistribution, self).__init__()
        self.p_conditional_distribution = p_conditional_distribution
        #Bottom levels dims actually works on double resolution of first set of independent pixels.
        #This is a bit of an implementation artefact of us reusing parallelcnn's for generating first level
        self.bottom_level_dims = 2*dims[1]//(2**levels)
        self.parallelcnns = nn.ModuleList( create_parallelcnns( [ dims[0], self.bottom_level_dims, self.bottom_level_dims ], p_conditional_distribution.params_size( dims[0] ), parallel_cnn_fn ) )
        self.pixel_channel_groups = nn.Parameter( torch.tensor( generate_pixel_channel_groups( [ dims[0], self.bottom_level_dims, self.bottom_level_dims ] ).astype( np.float32 ) ), requires_grad = False )
        self.information_masks = nn.Parameter( torch.tensor( generate_information_masks( [ dims[0], self.bottom_level_dims, self.bottom_level_dims ] ).astype( np.float32 ) ), requires_grad = False )
        self.upsamplers = nn.ModuleList( [ Upsampler(  [ dims[0], dims[1]//(2**(levels-level)), dims[1]//(2**(levels-level)) ], p_conditional_distribution, parallel_cnn_fn ) for level in range( levels ) ] )
        self.dims = dims
        self.levels = levels

    def log_prob( self, samples, conditional_inputs ):
        subsamples = samples[:,:,::2**(self.levels-1),::2**(self.levels-1)]
        subconditional_inputs = conditional_inputs[:,:,::2**(self.levels-1),::2**(self.levels-1)]
        
        output_log_prob = torch.tensor( np.zeros( subsamples.shape[0] ) ).to( subsamples.device )
        for n in range( self.dims[0] ):
            masked_input = subsamples*self.information_masks[n]
            subnet_input = torch.cat( (
                masked_input,
                subconditional_inputs ), dim=1 )
            subnet_output_logits = self.parallelcnns[n]( subnet_input )
            toutput_log_prob = self.p_conditional_distribution.log_prob( subsamples, subnet_output_logits, mask = self.pixel_channel_groups[n] )
            output_log_prob += toutput_log_prob
        
        upsampled_log_prob = output_log_prob
        for level in range( self.levels ):
            upsampled_log_prob += self.upsamplers[ level ].log_prob( samples[:,:,::2**(self.levels-level-1),::2**(self.levels-level-1)], conditional_inputs[:,:,::2**(self.levels-level-1),::2**(self.levels-level-1)] )

        return upsampled_log_prob

    def sample( self, conditional_inputs ):
        subsamples = torch.tensor( np.zeros( [ conditional_inputs.shape[0], self.dims[0], self.bottom_level_dims, self.bottom_level_dims ] ).astype( np.float32 ) ).to( conditional_inputs.device )
        subconditional_inputs = conditional_inputs[:,:,::2**(self.levels-1),::2**(self.levels-1)]
        
        for n in range( self.dims[0] ):
            subnet_input = torch.cat(
                ( subsamples*self.information_masks[n],
                subconditional_inputs ), dim=1 )
            subnet_output_logits = self.parallelcnns[n]( subnet_input )

            subsamples += self.p_conditional_distribution.sample( subnet_output_logits ) * \
                self.pixel_channel_groups[n]

        u = self.upsamplers[0].sample( subsamples[:,:,::2,::2], conditional_inputs[:,:,::2**(self.levels-1),::2**(self.levels-1)] )
        for level in range( 1, self.levels ):
            u = self.upsamplers[ level ].sample( u, conditional_inputs[:,:,::2**(self.levels-level-1),::2**(self.levels-level-1)] )

        return u

    def params_size( self, channels ):
        return 1

activation_fn = F.relu

#Achieved epoch 82, training 3620, validation 3902 on celeba aligned 20,000 images, batch size 16, level = 4, quantized output distribution
class unet( nn.Module ):
    def __init__( self, dims, params_size ):
        super(unet, self).__init__()
        self.upconv1_1 = torch.nn.Conv2d( dims[0]+1,64,3, padding=1 )
        self.upconv1_2 = torch.nn.Conv2d( 64, 64, 3, padding=1)
        
        self.upconv2_1 = torch.nn.Conv2d( 64, 128, 3, padding = 1, stride = 2 )
        self.upconv2_2 = torch.nn.Conv2d( 128, 128, 3, padding=1)
        
        self.downconv1_3 = torch.nn.Conv2d( 128, 64, 3, padding = 1 )
        self.downconv1_2 = torch.nn.Conv2d( 64, 64, 3, padding = 1 )
        self.downconv1_1 = torch.nn.Conv2d( 64, params_size, 1, padding = 0 )
        
        self.downconv2_2 = torch.nn.Conv2d( 128, 128, 3, padding = 1 )
        self.downconv2_1 = torch.nn.Conv2d( 128, 128, 3, padding = 1 )
        self.downconv2_upsample = torch.nn.Conv2d( 128, 256, 1, padding = 0 )

    
    def forward( self, x ):
        x = self.upconv1_1( x )
        x = activation_fn( x )
        x = self.upconv1_2( x )
        c1 = activation_fn( x )
        
        x = self.upconv2_1( c1 )
        x = activation_fn( x )
        x = self.upconv2_2( x )
        c2 = activation_fn( x )
        
        x = self.downconv2_2( c2 )
        x = activation_fn( x )
        x = self.downconv2_1( x )
        x = activation_fn( x )
        x = self.downconv2_upsample( x )

        orig = x.shape
        x = x.permute(0, 2, 3, 1).view( orig[0], orig[2], orig[3], orig[1]//4, 2, 2)
        x = x.permute(0, 1, 4, 2, 5, 3 ).contiguous().view( orig[0], orig[2]*2, orig[3]*2, orig[1]//4 )
        x = x.permute( 0, 3, 1, 2 )
        
        x = torch.cat( [ c1, x ], axis = 1 )
        x = self.downconv1_3( x )
        x = activation_fn( x )
        x = self.downconv1_2( x )
        x = activation_fn( x )
        x = self.downconv1_1( x )
        
        return x

def unet_parallel_cnn_fn( dims, params_size ):
    return unet( dims, params_size )
