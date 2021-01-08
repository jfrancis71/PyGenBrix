import torch.nn as nn
import torch
import numpy as np
import pl_bolts.models.vision.unet as plt_unet

class UpsamplerDistribution( nn.Module ):
    def __init__( self, output_distribution, downsampled_images, distribution_params, parallelcnns ):
        super( UpsamplerDistribution, self ).__init__()
        self.distribution_params = distribution_params
        self.parallelcnns = parallelcnns
        self.output_distribution = output_distribution
        self.downsampled_images = downsampled_images
        self.device = downsampled_images.device
        self.num_channels = len( self.parallelcnns )

#Note this will alter allowed_information
    def log_prob_block( self, upsampled_images, allowed_information, slice_h, slice_w ):
        block_log_prob = 0.0
        for channel in range( self.num_channels ):
            network_input = torch.cat( ( allowed_information, self.distribution_params ), dim = 1 )
            output_distribution_params = self.parallelcnns[ channel ][ 0 ]( network_input )
            block_log_prob += self.output_distribution( output_distribution_params[:,:,1::2,1::2] ).log_prob( upsampled_images[:,channel:channel+1,1::2,1::2] )["log_prob"]
            allowed_information[:,channel,1::2,1::2] += upsampled_images[:,channel,1::2,1::2]
        return block_log_prob

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
        output_log_prob = 0.0
        allowed_information = 0.0 * upsampled_images
        allowed_information[:,:,::2,::2] += upsampled_images[:,:,::2,::2]

        #predict all odd pixels
        block_log_prob = self.log_prob_block( upsampled_images, allowed_information, slice(1,None,2), slice(1,None,2) )
        logging_dict["block1_log_prob"] = block_log_prob
        output_log_prob += block_log_prob

        #predict all pixels even row, odd column
        block_log_prob = self.log_prob_block( upsampled_images, allowed_information, slice(0,None,2), slice(1,None,2) )
        logging_dict["block2_log_prob"] = block_log_prob
        output_log_prob += block_log_prob

        #predict all pixels odd row, even column
        block_log_prob = self.log_prob_block( upsampled_images, allowed_information, slice(1,None,2), slice(0,None,2) )
        logging_dict["block3_log_prob"] = block_log_prob
        output_log_prob += block_log_prob

        logging_dict["log_prob"] = output_log_prob

        return logging_dict

#Note, alters samples
    def sample_block( self, samples, slice_h, slice_w ):
        for channel in range( self.num_channels ):
            network_input = torch.cat( ( samples, self.distribution_params ), dim = 1 )
            output_distribution_params = self.parallelcnns[ channel ][ 0 ]( network_input )
            samples[:,channel,slice_h,slice_w] = self.output_distribution( output_distribution_params ).sample()[:,0,slice_h,slice_w]

    def sample( self ):
        samples = torch.tensor( np.zeros( [ self.downsampled_images.shape[0], self.downsampled_images.shape[1], self.downsampled_images.shape[2]*2, self.downsampled_images.shape[3]*2 ] ).astype( np.float32 ) ).to( self.device )
        samples[:,:,::2,::2] += self.downsampled_images

        #predict all odd pixels
        self.sample_block( samples, slice( 1, None, 2), slice( 1, None, 2 ) )

        #predict all pixels even row, odd column
        self.sample_block( samples, slice( 0, None, 2), slice( 1, None, 2 ) )

        #predict all pixels odd row, even column
        self.sample_block( samples, slice( 1, None, 2), slice( 0, None, 2 ) )

        return samples

class MultiStageParallelCNNDistribution( nn.Module ):
    def __init__( self, output_distribution, dims, bottom_parallelcnns, upsample_parallelcnns, levels, distribution_params ):
        super( MultiStageParallelCNNDistribution, self ).__init__()
        self.output_distribution = output_distribution
        self.levels = levels
        self.distribution_params = distribution_params
#        self.bottom_level_dims = 2*dims[1]//(2**levels)
        self.bottom_parallelcnns = bottom_parallelcnns
        self.upsample_parallelcnns = upsample_parallelcnns
        self.dims = dims

    def log_prob( self, samples ):
        bottom_samples = samples[:,:,::2**self.levels,::2**self.levels]
        bottom_distribution_params = self.distribution_params[:,:,::2**self.levels,::2**self.levels]

        logging_dict = {}
        
        output_log_prob = 0.0
        allowed_information = 0.0 * bottom_samples
        no_channels = len( self.bottom_parallelcnns )
        #predict all even pixels
        for channel in range( no_channels ):
            network_input = torch.cat( ( allowed_information, bottom_distribution_params ), dim = 1 )
            output_distribution_params = self.bottom_parallelcnns[ channel ]( network_input )
            channel_log_prob = self.output_distribution( output_distribution_params ).log_prob( bottom_samples[:,channel:channel+1] )["log_prob"]
            logging_dict["base_log_prob/channel_"+str(channel)] = channel_log_prob
            output_log_prob += channel_log_prob
            allowed_information[:,channel] = bottom_samples[:,channel]
        logging_dict["base_log_prob"] = output_log_prob.clone()
            
        for level in range( self.levels ):
            upsample_log_prob_dict = UpsamplerDistribution(
                self.output_distribution,
                samples[:,:,::2**(self.levels-level),::2**(self.levels-level)],
                self.distribution_params[:,:,::2**(self.levels-level-1),::2**(self.levels-level-1)],
                self.upsample_parallelcnns[ level ] ).log_prob( samples[:,:,::2**(self.levels-level-1),::2**(self.levels-level-1)] )
            logging_dict["upsample_level_"+str(level)+"/block1_log_prob"] = upsample_log_prob_dict["block1_log_prob"]
            logging_dict["upsample_level_"+str(level)+"/block2_log_prob"] = upsample_log_prob_dict["block2_log_prob"]
            logging_dict["upsample_level_"+str(level)+"/block3_log_prob"] = upsample_log_prob_dict["block3_log_prob"]
            logging_dict["upsample_level_"+str(level)+"/log_prob"] = upsample_log_prob_dict["log_prob"]
            output_log_prob += upsample_log_prob_dict["log_prob"]

        logging_dict["log_prob"] = output_log_prob
            
        return logging_dict
    
    def sample( self ):
        no_channels = len( self.bottom_parallelcnns )
        sample = torch.zeros( [ self.distribution_params.shape[0], self.dims[0], self.dims[1]//2**self.levels, self.dims[2]//2**self.levels ] ).to( self.distribution_params.device )
        bottom_distribution_params = self.distribution_params[:,:,::2**self.levels,::2**self.levels]
        for channel in range( no_channels ):
            network_input = torch.cat( ( sample, bottom_distribution_params ), dim = 1 )
            output_distribution_params = self.bottom_parallelcnns[ channel ]( network_input )
            tp = self.output_distribution( output_distribution_params ).sample()
            sample[0,channel] = tp[0,0]
            
        for level in range( self.levels ):
            sample = UpsamplerDistribution(
                self.output_distribution,
                sample,
                self.distribution_params[:,:,::2**(self.levels-level-1),::2**(self.levels-level-1)],
                self.upsample_parallelcnns[ level ] ).sample()
            
        return sample

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

class MultiStageParallelCNNLayer( nn.Module ):
    def __init__( self, dims, output_distribution, upsampling_stages, parallel_cnn_fn = default_parallel_cnn_fn ):
        super( MultiStageParallelCNNLayer, self ).__init__()
        bottom_width = dims[1]/2**upsampling_stages
        num_layers = int( min( bottom_width - 1, 3 ) )
        num_distribution_params = output_distribution.params_size( 1 )
        self.bottom_net = nn.ModuleList( [ 
        #Bug in PLT.UNet for num_layers = 0
            ( plt_unet.UNet( num_classes = num_distribution_params, input_channels = 1 + dims[0], num_layers = num_layers ) if num_layers > 0 else default_parallel_cnn_fn( dims, num_distribution_params ) ) for c in range( dims[0] ) ] )
        upsampler_nets = []
        for l in range( upsampling_stages ):
            output_width = bottom_width * 2**(l+1)
            num_layers = int( min( output_width - 1, 3 ) )
            upsampler_nets.append(
                nn.ModuleList( [
                    nn.ModuleList( [ plt_unet.UNet( num_distribution_params, input_channels = 1 + dims[0], num_layers = num_layers ) for s in range(3) ] ) for c in range(dims[0]) ] ) )
        self.upsampling_stages = upsampling_stages
        self.output_distribution = output_distribution
        self.dims = dims
        self.upsampler_nets = nn.ModuleList( upsampler_nets )
    
    def forward( self, distribution_params ):
        return MultiStageParallelCNNDistribution( self.output_distribution, self.dims, self.bottom_net, self.upsampler_nets, self.upsampling_stages, distribution_params )

    def params_size( self, channels ):
        return 1
