import math
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
    def log_prob_block( self, upsampled_images, allowed_information, block, slice_h, slice_w ):
        block_log_prob = 0.0
        for channel in range( self.num_channels ):
            network_input = torch.cat( ( allowed_information, self.distribution_params ), dim = 1 )
            output_distribution_params = self.parallelcnns[ channel ][ block ]( network_input )
            block_log_prob += self.output_distribution( output_distribution_params[:,:,slice_h,slice_w] ).log_prob( upsampled_images[:,channel:channel+1,slice_h,slice_w] )["log_prob"]
            allowed_information[:,channel,slice_h,slice_w] = upsampled_images[:,channel,slice_h,slice_w]
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
        log_prob = 0.0
        allowed_information = 0.0 * upsampled_images
        allowed_information[:,:,::2,::2] = upsampled_images[:,:,::2,::2]

        #predict all odd pixels
        block_log_prob = self.log_prob_block( upsampled_images, allowed_information, 0, slice(1,None,2), slice(1,None,2) )
        logging_dict["block1_log_prob"] = block_log_prob
        log_prob += block_log_prob

        #predict all pixels even row, odd column
        block_log_prob = self.log_prob_block( upsampled_images, allowed_information, 1, slice(0,None,2), slice(1,None,2) )
        logging_dict["block2_log_prob"] = block_log_prob
        log_prob += block_log_prob

        #predict all pixels odd row, even column
        block_log_prob = self.log_prob_block( upsampled_images, allowed_information, 2, slice(1,None,2), slice(0,None,2) )
        logging_dict["block3_log_prob"] = block_log_prob
        log_prob += block_log_prob

        logging_dict["log_prob"] = log_prob

        return logging_dict

#Note, alters samples
    def sample_block( self, samples, block, slice_h, slice_w ):
        for channel in range( self.num_channels ):
            network_input = torch.cat( ( samples, self.distribution_params ), dim = 1 )
            output_distribution_params = self.parallelcnns[ channel ][ block ]( network_input )
            samples[:,channel,slice_h,slice_w] = self.output_distribution( output_distribution_params ).sample()[:,0,slice_h,slice_w]

    def sample( self ):
        samples = torch.zeros( [ self.downsampled_images.shape[0], self.downsampled_images.shape[1], self.downsampled_images.shape[2]*2, self.downsampled_images.shape[3]*2 ] ).to( self.device )
        samples[:,:,::2,::2] = self.downsampled_images

        #predict all odd pixels
        self.sample_block( samples, 0, slice( 1, None, 2), slice( 1, None, 2 ) )

        #predict all pixels even row, odd column
        self.sample_block( samples, 1, slice( 0, None, 2), slice( 1, None, 2 ) )

        #predict all pixels odd row, even column
        self.sample_block( samples, 2, slice( 1, None, 2), slice( 0, None, 2 ) )

        return samples

class MultiStageParallelCNNDistribution( nn.Module ):
    def __init__( self, output_distribution, dims, bottom_parallelcnns, upsample_parallelcnns, num_upsampling_stages, distribution_params ):
        super( MultiStageParallelCNNDistribution, self ).__init__()
        self.output_distribution = output_distribution
        self.num_upsampling_stages = num_upsampling_stages
        self.distribution_params = distribution_params
#        self.bottom_level_dims = 2*dims[1]//(2**num_upsampling_stages)
        self.bottom_parallelcnns = bottom_parallelcnns
        self.upsample_parallelcnns = upsample_parallelcnns
        self.dims = dims

    def log_prob( self, samples ):
        bottom_samples = samples[:,:,::2**self.num_upsampling_stages,::2**self.num_upsampling_stages]
        bottom_distribution_params = self.distribution_params[:,:,::2**self.num_upsampling_stages,::2**self.num_upsampling_stages]

        logging_dict = {}
        
        log_prob = 0.0
        allowed_information = 0.0 * bottom_samples
        num_channels = len( self.bottom_parallelcnns )
        #predict all even pixels
        for channel in range( num_channels ):
            network_input = torch.cat( ( allowed_information, bottom_distribution_params ), dim = 1 )
            output_distribution_params = self.bottom_parallelcnns[ channel ]( network_input )
            channel_log_prob = self.output_distribution( output_distribution_params ).log_prob( bottom_samples[:,channel:channel+1] )["log_prob"]
            logging_dict["base_log_prob/channel_"+str(channel)] = channel_log_prob
            log_prob += channel_log_prob
            allowed_information[:,channel] = bottom_samples[:,channel]
        logging_dict["base_log_prob"] = log_prob.clone()
            
        for level in range( self.num_upsampling_stages ):
            upsample_log_prob_dict = UpsamplerDistribution(
                self.output_distribution,
                samples[:,:,::2**(self.num_upsampling_stages-level),::2**(self.num_upsampling_stages-level)],
                self.distribution_params[:,:,::2**(self.num_upsampling_stages-level-1),::2**(self.num_upsampling_stages-level-1)],
                self.upsample_parallelcnns[ level ] ).log_prob( samples[:,:,::2**(self.num_upsampling_stages-level-1),::2**(self.num_upsampling_stages-level-1)] )
            logging_dict["upsample_level_"+str(level)+"/block1_log_prob"] = upsample_log_prob_dict["block1_log_prob"]
            logging_dict["upsample_level_"+str(level)+"/block2_log_prob"] = upsample_log_prob_dict["block2_log_prob"]
            logging_dict["upsample_level_"+str(level)+"/block3_log_prob"] = upsample_log_prob_dict["block3_log_prob"]
            logging_dict["upsample_level_"+str(level)+"/log_prob"] = upsample_log_prob_dict["log_prob"]
            log_prob += upsample_log_prob_dict["log_prob"]

        logging_dict["log_prob"] = log_prob
            
        return logging_dict
    
    def sample( self ):
        num_channels = len( self.bottom_parallelcnns )
        sample = torch.zeros( [ self.distribution_params.shape[0], self.dims[0], self.dims[1]//2**self.num_upsampling_stages, self.dims[2]//2**self.num_upsampling_stages ] ).to( self.distribution_params.device )
        bottom_distribution_params = self.distribution_params[:,:,::2**self.num_upsampling_stages,::2**self.num_upsampling_stages]
        for channel in range( num_channels ):
            network_input = torch.cat( ( sample, bottom_distribution_params ), dim = 1 )
            output_distribution_params = self.bottom_parallelcnns[ channel ]( network_input )
            tp = self.output_distribution( output_distribution_params ).sample()
            sample[0,channel] = tp[0,0]
            
        for level in range( self.num_upsampling_stages ):
            sample = UpsamplerDistribution(
                self.output_distribution,
                sample,
                self.distribution_params[:,:,::2**(self.num_upsampling_stages-level-1),::2**(self.num_upsampling_stages-level-1)],
                self.upsample_parallelcnns[ level ] ).sample()
            
        return sample

class MultiStageParallelCNNLayer( nn.Module ):
    def __init__( self, dims, output_distribution, num_upsampling_stages, max_unet_layers = 3 ):
        super( MultiStageParallelCNNLayer, self ).__init__()
        bottom_width = dims[1]/2**num_upsampling_stages
        num_distribution_params = output_distribution.params_size( 1 )
        unet_num_layers = int( min( math.log( bottom_width ) + 1, max_unet_layers ) )
        self.bottom_net = nn.ModuleList( [ 
            plt_unet.UNet( num_classes = num_distribution_params, input_channels = 1 + dims[0], num_layers = unet_num_layers ) for c in range( dims[0] ) ] )
        upsampler_nets = []
        for l in range( num_upsampling_stages ):
            output_width = bottom_width * 2**(l+1)
            unet_num_layers = int( min( math.log( output_width ) + 1, max_unet_layers ) )
            upsampler_nets.append(
                nn.ModuleList( [
                    nn.ModuleList( [ plt_unet.UNet( num_distribution_params, input_channels = 1 + dims[0], num_layers = unet_num_layers ) for s in range(3) ] ) for c in range(dims[0]) ] ) )
        self.num_upsampling_stages = num_upsampling_stages
        self.output_distribution = output_distribution
        self.dims = dims
        self.upsampler_nets = nn.ModuleList( upsampler_nets )
    
    def forward( self, distribution_params ):
        return MultiStageParallelCNNDistribution( self.output_distribution, self.dims, self.bottom_net, self.upsampler_nets, self.num_upsampling_stages, distribution_params )

    def params_size( self, channels ):
        return 1
