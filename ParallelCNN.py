import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class UpsamplerDistribution( nn.Module ):
    def __init__( self, output_distribution, input_low_res_image, logits, parallelcnns ):
        super( UpsamplerDistribution, self ).__init__()
        self.logits = logits
        self.parallelcnns = parallelcnns
        self.output_distribution = output_distribution
        self.input_low_res_image = input_low_res_image
        self.device = input_low_res_image.device

#Compute the log prob of samples conditioned on even pixels (where pixels counts from 0)
#but excluding the log prob of the even pixels themselves
#note samples will have double the spatial resolution of input_low_res_image
    def log_prob( self, samples ):
        assert( 2*self.input_low_res_image.shape[3] == samples.shape[3] )
        if (samples.shape[0] != self.logits.shape[0]):
            raise TypeError("samples batch size {}, but logits has batch size {}"
                            .format( samples.shape[0], self.logits.shape[0] ) )
        if (samples.shape[2:4] != self.logits.shape[2:4]):
            raise TypeError("samples spatial shape  {}, but logits has spatial shape {}"
                            .format( samples.shape[2:4], self.logits.shape[2:4] ) )
        if ( self.logits.shape[1] != 1 ):#MultiStateParallelConditionalCNN assumes logits has channel size 1
            raise TypeError("conditional_inputs has channel size {}, but should be 1"
                            .format( self.logits.shape[1] ) )
        if ( samples[0,0,0,0] != self.input_low_res_image[0,0,0,0] ):
            raise TypeError("The low res image doesn't appear to be the subsampled input sample")

        output_log_prob = 0.0
        allowed_information = 0.0 * samples
        allowed_information[:,:,::2,::2] += samples[:,:,::2,::2]
        no_channels = len( self.parallelcnns )
        #predict all odd pixels
        for channel in range( no_channels ):
            network_input = torch.cat( ( allowed_information, self.logits ), dim = 1 )
            network_output_logits = self.parallelcnns[ channel ][ 0 ]( network_input )
            output_log_prob += self.output_distribution( network_output_logits[:,:,1::2,1::2] ).log_prob( samples[:,channel::channel+1,1::2,1::2] )
            allowed_information[:,channel,1::2,1::2] += samples[:,channel,1::2,1::2]
        #predict all pixels even row, odd column
        for channel in range( no_channels ):
            network_input = torch.cat( ( allowed_information, self.logits ), dim = 1 )
            network_output_logits = self.parallelcnns[ channel ][ 1 ]( network_input )
            output_log_prob += self.output_distribution( network_output_logits[:,:,::2,1::2] ).log_prob( samples[:,channel::channel+1,::2,1::2] )
            allowed_information[:,channel,::2,1::2] += samples[:,channel,::2,1::2]
        #predict all pixels odd row, even column
        for channel in range( no_channels ):
            network_input = torch.cat( ( allowed_information, self.logits ), dim = 1 )
            network_output_logits = self.parallelcnns[ channel ][ 2 ]( network_input )
            output_log_prob += self.output_distribution( network_output_logits[:,:,1::2,::2] ).log_prob( samples[:,channel::channel+1,1::2,::2] )
            allowed_information[:,channel,1::2,::2] += samples[:,channel,1::2,::2]

        return output_log_prob

    def sample( self ):
        samples = torch.tensor( np.zeros( [ self.input_low_res_image.shape[0], self.input_low_res_image.shape[1], self.input_low_res_image.shape[2]*2, self.input_low_res_image.shape[3]*2 ] ).astype( np.float32 ) ).to( self.device )
        samples[:,:,::2,::2] += self.input_low_res_image
        no_channels = len( self.parallelcnns )
        #predict all odd pixels
        for channel in range( no_channels ):
            network_input = torch.cat( ( samples, self.logits ), dim = 1 )
            network_output_logits = self.parallelcnns[ channel ][ 0 ]( network_input )
            samples[:,channel,1::2,1::2] = self.output_distribution( network_output_logits ).sample()[:,0,1::2,1::2]
        #predict all pixels even row, odd column
        for channel in range( no_channels ):
            network_input = torch.cat( ( samples, self.logits ), dim = 1 )
            network_output_logits = self.parallelcnns[ channel ][ 1 ]( network_input )
            samples[:,channel,::2,1::2] = self.output_distribution( network_output_logits ).sample()[:,0,::2,1::2]
        #predict all pixels odd row, even column
        for channel in range( no_channels ):
            network_input = torch.cat( ( samples, self.logits ), dim = 1 )
            network_output_logits = self.parallelcnns[ channel ][ 2 ]( network_input )
            samples[:,channel,1::2,::2] = self.output_distribution( network_output_logits ).sample()[:,0,1::2,::2]

        return samples

class MultiStageParallelCNNDistribution( nn.Module ):
    def __init__( self, output_distribution, dims, bottom_parallelcnns, upsample_parallelcnns, levels, logits ):
        super( MultiStageParallelCNNDistribution, self ).__init__()
        self.output_distribution = output_distribution
        self.levels = levels
        self.logits = logits
#        self.bottom_level_dims = 2*dims[1]//(2**levels)
        self.bottom_parallelcnns = bottom_parallelcnns
        self.upsample_parallelcnns = upsample_parallelcnns
        self.dims = dims

    def log_prob( self, samples ):
        bottom_samples = samples[:,:,::2**self.levels,::2**self.levels]
        bottom_logit_inputs = self.logits[:,:,::2**self.levels,::2**self.levels]
        
        output_log_prob = 0.0
        allowed_information = 0.0 * bottom_samples
        no_channels = len( self.bottom_parallelcnns )
        #predict all even pixels
        for channel in range( no_channels ):
            network_input = torch.cat( ( allowed_information, bottom_logit_inputs ), dim = 1 )
            network_output_logits = self.bottom_parallelcnns[ channel ]( network_input )
            output_log_prob += self.output_distribution( network_output_logits ).log_prob( bottom_samples[:,channel:channel+1] )
            allowed_information[:,channel] = bottom_samples[:,channel]
            
        for level in range( self.levels ):
            output_log_prob += UpsamplerDistribution(
                self.output_distribution,
                samples[:,:,::2**(self.levels-level),::2**(self.levels-level)],
                self.logits[:,:,::2**(self.levels-level-1),::2**(self.levels-level-1)],
                self.upsample_parallelcnns[ level ] ).log_prob( samples[:,:,::2**(self.levels-level-1),::2**(self.levels-level-1)] )
            
        return output_log_prob
    
    def sample( self ):
        no_channels = len( self.bottom_parallelcnns )
        sample = torch.zeros( [ self.logits.shape[0], self.dims[0], self.dims[1]//2**self.levels, self.dims[2]//2**self.levels ] ).to( self.logits.device )
        bottom_logit_inputs = self.logits[:,:,::2**self.levels,::2**self.levels]
        for channel in range( no_channels ):
            network_input = torch.cat( ( sample, bottom_logit_inputs ), dim = 1 )
            network_output_logits = self.bottom_parallelcnns[ channel ]( network_input )
            tp = self.output_distribution( network_output_logits ).sample()
            sample[0,channel] = tp[0,0]
            
        for level in range( self.levels ):
            sample = UpsamplerDistribution(
                self.output_distribution,
                sample,
                self.logits[:,:,::2**(self.levels-level-1),::2**(self.levels-level-1)],
                self.upsample_parallelcnns[ level ] ).sample()
            
        return sample

class MultiStageParallelCNNLayer( nn.Module ):
    def __init__( self, dims, output_distribution, levels ):
        super( MultiStageParallelCNNLayer, self ).__init__()
        self.bottom_pcnn = nn.ModuleList( [ default_parallel_cnn_fn( [ dims[0], 1024, 1024 ], output_distribution.params_size( 1 ) ) for s in range( dims[0] ) ] )
        self.upsamplers_nets = nn.ModuleList( [ nn.ModuleList( [ nn.ModuleList( [ default_parallel_cnn_fn( [ dims[0], 1024, 1024 ], output_distribution.params_size( 1 ) ) for s in range(3) ] ) for c in range(dims[0]) ] ) for l in range(levels) ] )
        self.levels = levels
        self.output_distribution = output_distribution
        self.dims = dims
    
    def forward( self, logits ):
        return MultiStageParallelCNNDistribution( self.output_distribution, self.dims, self.bottom_pcnn, self.upsamplers_nets, self.levels, logits )

    #These should be changed to support new distribution style
    def log_prob( self, sample, conditionals ):
        return self.forward( conditionals ).log_prob( sample )
    
    def sample( self, conditionals ):
        return self.forward( conditionals ).sample()

    def params_size( self, channels ):
        return 1

#Achieved epoch 10, training 6606, validation 6725 on celeba aligned 100,000 images, batch size 8, level = 1, quantized output distribution
#Achieved epoch 10, training 4167, validation 4059 on celeba aligned 100,000 images, batch size 8, level = 4, quantized output distribution
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

activation_fn = F.relu

#Achieved epoch 10, training 3677, validation 3668 on celeba aligned 100,000 images, batch size 8, level = 4, quantized output distribution
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

#Achieved epoch 10, training 3474, validation 3522 on celeba aligned 100,000 images, batch size 8, level = 4, quantized output distribution
#Achieved epoch 10, training 3511, validation 3678 on celeba aligned 100,000 images, batch size 8, level = 4, quantized output distribution, no conditionals = 16
#Achieved epoch 10, training 3276, validation 3570 on celeba aligned 100,000 images, batch size 8, level = 4, quantized output distribution, activation=tanh
#Achieved epoch 53, training 2817, validation 4364 on celeba aligned 100,000 images, batch size 8, level = 4, quantized output distribution, clearly overtrained, but interesting
#Achieved epoch 10, training 3580, validation 3694 on coco cropped 100,000 images, batch size 8, level = 4, quantized output distribution
class unet1( nn.Module ):
    def __init__( self, dims, params_size ):
        super(unet1, self).__init__()
        self.upconv1_1 = torch.nn.Conv2d( dims[0]+1,64,3, padding=1 )
        self.upconv1_2 = torch.nn.Conv2d( 64, 64, 3, padding=1)

        self.upconv2_1 = torch.nn.Conv2d( 64, 128, 3, padding = 1, stride = 2 )
        self.upconv2_2 = torch.nn.Conv2d( 128, 128, 3, padding=1)
        
        self.upconv3_1 = torch.nn.Conv2d( 128, 256, 3, padding = 1, stride = 2 )
        self.upconv3_2 = torch.nn.Conv2d( 256, 256, 3, padding=1)

        self.downconv1_3 = torch.nn.Conv2d( 128, 64, 3, padding = 1 )
        self.downconv1_2 = torch.nn.Conv2d( 64, 64, 3, padding = 1 )
        self.downconv1_1 = torch.nn.Conv2d( 64, params_size, 1, padding = 0 )

        self.downconv2_2 = torch.nn.Conv2d( 256, 128, 3, padding = 1 )
        self.downconv2_1 = torch.nn.Conv2d( 128, 128, 3, padding = 1 )
        self.downconv2_upsample = torch.nn.Conv2d( 128, 256, 1, padding = 0 )
        
        self.downconv3_2 = torch.nn.Conv2d( 256, 256, 3, padding = 1 )
        self.downconv3_1 = torch.nn.Conv2d( 256, 256, 3, padding = 1 )
        self.downconv3_upsample = torch.nn.Conv2d( 256, 512, 1, padding = 0 )


    def forward( self, x ):
        x = self.upconv1_1( x )
        x = activation_fn( x )
        x = self.upconv1_2( x )
        c1 = activation_fn( x )

        x = self.upconv2_1( c1 )
        x = activation_fn( x )
        x = self.upconv2_2( x )
        c2 = activation_fn( x )
        
        x = self.upconv3_1( c2 )
        x = activation_fn( x )
        x = self.upconv3_2( x )
        c3 = activation_fn( x )

        x = self.downconv3_2( c3 )
        x = activation_fn( x )
        x = self.downconv3_1( x )
        x = activation_fn( x )
        x = self.downconv3_upsample( x )

        orig = x.shape
        x = x.permute(0, 2, 3, 1).view( orig[0], orig[2], orig[3], orig[1]//4, 2, 2)
        x = x.permute(0, 1, 4, 2, 5, 3 ).contiguous().view( orig[0], orig[2]*2, orig[3]*2, orig[1]//4 )
        x = x.permute( 0, 3, 1, 2 )

        x = torch.cat( [ c2, x ], axis = 1 )
        x = self.downconv2_2( x )
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

def unet_parallel_cnn_fn1( dims, params_size ):
    return unet1( dims, params_size )
