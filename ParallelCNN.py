import math
import torch.nn as nn
import torch
import numpy as np
import pl_bolts.models.vision.unet as plt_unet

base_slice = ( slice( 0, None, 2), slice( 0, None, 2 ) )
upsampling_slices = [ ( slice(1,None,2), slice(1,None,2) ),
    ( slice(0,None,2), slice(1,None,2) ),
    ( slice( 1, None, 2), slice( 0, None, 2 ) ) ]


#Note this will alter masked_value
def log_prob_block( upsampled_images, output_distribution, distribution_params, masked_value, block_parallel_cnns, slice ):
    block_log_prob = 0.0
    for channel in range( len( block_parallel_cnns ) ):
        network_input = torch.cat( ( masked_value, distribution_params ), dim = 1 )
        output_distribution_params = block_parallel_cnns[ channel ]( network_input )
        block_log_prob += output_distribution( output_distribution_params[:,:,slice[0],slice[1]] ).log_prob( upsampled_images[:,channel:channel+1,slice[0],slice[1]] )["log_prob"]
        masked_value[:,channel,slice[0],slice[1]] = upsampled_images[:,channel,slice[0],slice[1]]
    return block_log_prob

#Note, alters samples
def sample_block( samples, output_distribution, distribution_params, block_parallel_cnns, slice ):
    for channel in range( len( block_parallel_cnns ) ):
        network_input = torch.cat( ( samples, distribution_params ), dim = 1 )
        output_distribution_params = block_parallel_cnns[ channel ]( network_input )
        samples[:,channel,slice[0],slice[1]] = output_distribution( output_distribution_params ).sample()[:,0,slice[0],slice[1]]

class UpsamplerDistribution( nn.Module ):
    def __init__( self, output_distribution, downsampled_images, distribution_params, parallel_nets ):
        super( UpsamplerDistribution, self ).__init__()
        self.distribution_params = distribution_params
        self.parallel_nets = parallel_nets
        self.output_distribution = output_distribution
        self.downsampled_images = downsampled_images
        self.device = downsampled_images.device

#Compute the log prob of samples conditioned on even pixels (where pixels counts from 0)
#but excluding the log prob of the even pixels themselves
#note samples will have double the spatial resolution of input_low_res_image
    def log_prob( self, upsampled_images ):
        assert( 2*self.downsampled_images.shape[3] == upsampled_images.shape[3] )
        if (upsampled_images.shape[0] != self.distribution_params.shape[0]):
            raise TypeError("samples batch size {}, but logits has batch size {}"
                            .format( upsampled_images.shape[0], self.distribution_params.shape[0] ) )
        if (upsampled_images.shape[2:4] != self.distribution_params.shape[2:4]):
            raise TypeError("upsampled_images spatial shape  {}, but distribution_params has spatial shape {}"
                            .format( upsampled_images.shape[2:4], self.distribution_params.shape[2:4] ) )
        if ( self.distribution_params.shape[1] != 1 ):#MultiStateParallelConditionalCNN assumes logits has channel size 1
            raise TypeError("distribution_params has channel size {}, but should be 1"
                            .format( self.distribution_params.shape[1] ) )
        if ( upsampled_images[0,0,0,0] != self.downsampled_images[0,0,0,0] ):
            raise TypeError("The downsampled image doesn't appear to be the subsampled upsampled image")

        logging_dict = {}
        masked_value = torch.zeros_like( upsampled_images )
        masked_value[:,:,base_slice[0],base_slice[1]] = upsampled_images[:,:,base_slice[0],base_slice[1]]

        log_prob = 0.0
        for s in range(3):
            block_log_prob = log_prob_block( upsampled_images, self.output_distribution, self.distribution_params, masked_value, self.parallel_nets[s], upsampling_slices[s] )
            logging_dict["block"+str(s)+"_log_prob"] = block_log_prob
            log_prob += block_log_prob

        logging_dict["log_prob"] = log_prob

        return logging_dict

    def sample( self ):
        samples = torch.zeros( [ self.downsampled_images.shape[0], self.downsampled_images.shape[1], self.downsampled_images.shape[2]*2, self.downsampled_images.shape[3]*2 ], device = self.device )
        samples[:,:,base_slice[0],base_slice[1]] = self.downsampled_images

        for s in range(3):
            sample_block( samples, self.output_distribution, self.distribution_params, self.parallel_nets[s], upsampling_slices[s] )


        return samples

class ParallelCNNDistribution( nn.Module ):
    def __init__( self, output_distribution, dims, base_parallel_nets, upsample_parallel_nets, distribution_params ):
        super( ParallelCNNDistribution, self ).__init__()
        self.output_distribution = output_distribution
        self.num_upsampling_stages = len( upsample_parallel_nets, )
        self.distribution_params = distribution_params
        self.base_parallel_nets = base_parallel_nets
        self.upsample_parallel_nets = upsample_parallel_nets
        self.dims = dims

    def log_prob( self, value ):
        base_samples = value[:,:,::2**self.num_upsampling_stages,::2**self.num_upsampling_stages]
        base_distribution_params = self.distribution_params[:,:,::2**self.num_upsampling_stages,::2**self.num_upsampling_stages]

        logging_dict = {}
        
        log_prob = 0.0
        masked_value = torch.zeros_like( base_samples )

        #predict all even pixels
        log_prob = log_prob_block( base_samples, self.output_distribution, base_distribution_params, masked_value, self.base_parallel_nets, base_slice )
        logging_dict["base_log_prob"] = log_prob.clone()
            
        for level in range( self.num_upsampling_stages ):
            upsample_log_prob_dict = UpsamplerDistribution(
                self.output_distribution,
                value[:,:,::2**(self.num_upsampling_stages-level),::2**(self.num_upsampling_stages-level)],
                self.distribution_params[:,:,::2**(self.num_upsampling_stages-level-1),::2**(self.num_upsampling_stages-level-1)],
                self.upsample_parallel_nets[ level ] ).log_prob( value[:,:,::2**(self.num_upsampling_stages-level-1),::2**(self.num_upsampling_stages-level-1)] )
            for s in range(3):
                logging_dict["upsample_level_"+str(level)+"/block"+str(s)+"_log_prob"] = upsample_log_prob_dict["block"+str(s)+"_log_prob"]
            logging_dict["upsample_level_"+str(level)+"/log_prob"] = upsample_log_prob_dict["log_prob"]
            log_prob += upsample_log_prob_dict["log_prob"]

        logging_dict["log_prob"] = log_prob
            
        return logging_dict
    
    def sample( self ):
        sample = torch.zeros( [ self.distribution_params.shape[0], self.dims[0], self.dims[1]//2**self.num_upsampling_stages, self.dims[2]//2**self.num_upsampling_stages ], device = self.distribution_params.device )
        base_distribution_params = self.distribution_params[:,:,::2**self.num_upsampling_stages,::2**self.num_upsampling_stages]
        sample_block( sample, self.output_distribution, base_distribution_params, self.base_parallel_nets, base_slice )
            
        for level in range( self.num_upsampling_stages ):
            sample = UpsamplerDistribution(
                self.output_distribution,
                sample,
                self.distribution_params[:,:,::2**(self.num_upsampling_stages-level-1),::2**(self.num_upsampling_stages-level-1)],
                self.upsample_parallel_nets[ level ] ).sample()
            
        return sample

class ParallelCNNLayer( nn.Module ):
    def __init__( self, dims, output_distribution, num_upsampling_stages, max_unet_layers = 3 ):
        super( ParallelCNNLayer, self ).__init__()
        base_width = dims[1]/2**num_upsampling_stages
        num_distribution_params = output_distribution.params_size( 1 )
        unet_num_layers = int( min( math.log( base_width ) + 1, max_unet_layers ) )
        self.base_nets = nn.ModuleList( [ 
            plt_unet.UNet( num_classes = num_distribution_params, input_channels = 1 + dims[0], num_layers = unet_num_layers ) for c in range( dims[0] ) ] )
        upsampler_nets = []
        for l in range( num_upsampling_stages ):
            output_width = base_width * 2**(l+1)
            unet_num_layers = int( min( math.log( output_width ) + 1, max_unet_layers ) )
            upsampler_nets.append(
                nn.ModuleList( [
                    nn.ModuleList( [ plt_unet.UNet( num_distribution_params, input_channels = 1 + dims[0], num_layers = unet_num_layers ) for c in range(dims[0]) ] ) for s in range(3) ] ) )
        self.num_upsampling_stages = num_upsampling_stages
        self.output_distribution = output_distribution
        self.dims = dims
        self.upsampler_nets = nn.ModuleList( upsampler_nets )
    
    def forward( self, x ):
        return ParallelCNNDistribution( self.output_distribution, self.dims, self.base_nets, self.upsampler_nets, x )

    def params_size( self, channels ):
        return 1
