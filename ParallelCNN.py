import math
import torch.nn as nn
import torch
import numpy as np
import pl_bolts.models.vision.unet as plt_unet

base_slice = ( slice( 0, None, 2), slice( 0, None, 2 ) )
upsampling_slices = [ ( slice(1,None,2), slice(1,None,2) ),
    ( slice(0,None,2), slice(1,None,2) ),
    ( slice( 1, None, 2), slice( 0, None, 2 ) ) ]

def delete_batch_norm_unet( unet ):
    del unet.layers[0].net[1]
    del unet.layers[0].net[3]
    for l in range( 1, len( unet.layers )//2 ):
        del unet.layers[l].net[1].net[1]
        del unet.layers[l].net[1].net[3]
    for l in range( (len( unet.layers )+1)//2, len( unet.layers )-1 ):
        del unet.layers[l].conv.net[1]
        del unet.layers[l].conv.net[3]       


class ParallelCNNDistribution( nn.Module ):
    def __init__( self, output_distribution, event_shape, base_parallel_nets, upsample_parallel_nets, distribution_params ):
        super( ParallelCNNDistribution, self ).__init__()
        self.output_distribution = output_distribution
        self.num_upsampling_stages = len( upsample_parallel_nets, )
        self.distribution_params = distribution_params
        self.base_parallel_nets = base_parallel_nets
        self.upsample_parallel_nets = upsample_parallel_nets
        self.event_shape = event_shape


#Note this will alter masked_value
    def log_prob_block( self, upsampled_images, distribution_params, masked_value, block_parallel_cnns, slice ):
        block_log_prob = 0.0
        for channel in range( len( block_parallel_cnns ) ):
            network_input = torch.cat( ( masked_value, distribution_params ), dim = 1 )
            output_distribution_params = block_parallel_cnns[ channel ]( network_input )
            block_log_prob += self.output_distribution( output_distribution_params[:,:,slice[0],slice[1]] ).log_prob( upsampled_images[:,channel:channel+1,slice[0],slice[1]] )["log_prob"]
            masked_value[:,channel,slice[0],slice[1]] = upsampled_images[:,channel,slice[0],slice[1]]
        return block_log_prob

#Note, alters samples
    def sample_block( self, samples, distribution_params, block_parallel_cnns, slice ):
        for channel in range( len( block_parallel_cnns ) ):
            network_input = torch.cat( ( samples, distribution_params ), dim = 1 )
            output_distribution_params = block_parallel_cnns[ channel ]( network_input )
            samples[:,channel,slice[0],slice[1]] = self.output_distribution( output_distribution_params ).sample()[:,0,slice[0],slice[1]]

    def upsampler_log_prob( self, value, distribution_params, parallel_cnns ):

        logging_dict = {}
        masked_value = torch.zeros_like( value )
        masked_value[:,:,base_slice[0],base_slice[1]] = value[:,:,base_slice[0],base_slice[1]]
        log_prob = 0.0
        for s in range(3):
            block_log_prob = self.log_prob_block( value, distribution_params, masked_value, parallel_cnns[s], upsampling_slices[s] )
            logging_dict["block"+str(s)+"_log_prob"] = block_log_prob
            log_prob += block_log_prob

        logging_dict["log_prob"] = log_prob

        return logging_dict

    def upsampler_sample( self, downsampled_images, distribution_params, parallel_nets ):
        samples = torch.zeros( [ downsampled_images.shape[0], downsampled_images.shape[1], downsampled_images.shape[2]*2, downsampled_images.shape[3]*2 ], device = downsampled_images.device )
        samples[:,:,base_slice[0],base_slice[1]] = downsampled_images

        for s in range(3):
            self.sample_block( samples, distribution_params, parallel_nets[s], upsampling_slices[s] )

        return samples


    def log_prob( self, value ):
        if (value.size()[0] != self.distribution_params.size()[0]):
            raise RuntimeError("value batch size {}, but distribution_params has batch size {}"
                            .format( value.shape[0], self.distribution_params.shape[0] ) )
        if (value.size()[1:4] != torch.Size( self.event_shape) ):
            raise RuntimeError("value shape  {}, but event_shape has shape {}"
                            .format( value.shape[1:4], self.event_shape ) )
        base_samples = value[:,:,::2**self.num_upsampling_stages,::2**self.num_upsampling_stages]
        base_distribution_params = self.distribution_params[:,:,::2**self.num_upsampling_stages,::2**self.num_upsampling_stages]

        logging_dict = {}
        
        log_prob = 0.0
        masked_value = torch.zeros_like( base_samples )

        #predict all base pixels
        log_prob = self.log_prob_block( base_samples, base_distribution_params, masked_value, self.base_parallel_nets, base_slice )
        logging_dict["base_log_prob"] = log_prob.clone()
            
        for level in range( self.num_upsampling_stages ):
            level_subsample_rate = 2**(self.num_upsampling_stages-level - 1)
            upsample_log_prob_dict = self.upsampler_log_prob(
                value[:,:,::level_subsample_rate,::level_subsample_rate],
                self.distribution_params[:,:,::level_subsample_rate,::level_subsample_rate],
                self.upsample_parallel_nets[ level ] )

            for k, v in upsample_log_prob_dict.items():
                logging_dict["upsample_level_"+str(level)+"/"+k] = upsample_log_prob_dict[k]

            log_prob += upsample_log_prob_dict["log_prob"]

        logging_dict["log_prob"] = log_prob
            
        return logging_dict
    
    def sample( self ):
        sample = torch.zeros( [ self.distribution_params.shape[0], self.event_shape[0], self.event_shape[1]//2**self.num_upsampling_stages, self.event_shape[2]//2**self.num_upsampling_stages ], device = self.distribution_params.device )
        base_distribution_params = self.distribution_params[:,:,::2**self.num_upsampling_stages,::2**self.num_upsampling_stages]
        self.sample_block( sample, base_distribution_params, self.base_parallel_nets, base_slice )
            
        for level in range( self.num_upsampling_stages ):

            level_subsample_rate = 2**(self.num_upsampling_stages-level - 1)
            sample = self.upsampler_sample(
                sample,
                self.distribution_params[:,:,::level_subsample_rate,::level_subsample_rate],
                self.upsample_parallel_nets[ level ] )
            
        return sample

class ParallelCNNLayer( nn.Module ):
    def __init__( self, event_shape, output_distribution, num_upsampling_stages, max_unet_layers = 3, params_size = 1, add_ones = True, batch = True ):
        super( ParallelCNNLayer, self ).__init__()
        base_width = event_shape[1]/2**num_upsampling_stages
        num_distribution_params = output_distribution.params_size( 1 )
        unet_num_layers = int( min( math.log( base_width ) + 1, max_unet_layers ) )
        input_channels = params_size + event_shape[0]
        if add_ones:
            input_channels += 1
        self.base_nets = nn.ModuleList( [ 
            plt_unet.UNet( num_classes = num_distribution_params, input_channels = input_channels, num_layers = unet_num_layers ) for c in range( event_shape[0] ) ] )
        upsampler_nets = []
        for l in range( num_upsampling_stages ):
            output_width = base_width * 2**(l+1)
            unet_num_layers = int( min( math.log( output_width ) + 1, max_unet_layers ) )
            upsampler_nets.append(
                nn.ModuleList( [
                    nn.ModuleList( [ plt_unet.UNet( num_distribution_params, input_channels = input_channels, num_layers = unet_num_layers ) for c in range(event_shape[0]) ] ) for s in range(3) ] ) )
        self.num_upsampling_stages = num_upsampling_stages
        self.output_distribution = output_distribution
        self.upsampler_nets = nn.ModuleList( upsampler_nets )
        self.event_shape = event_shape
        self.params_size = params_size
        self.add_ones = add_ones
        if batch == False:
            for c in range(event_shape[0]):
                delete_batch_norm_unet( self.base_nets[c] )
            for l in range( num_upsampling_stages ):
                for c in range( event_shape[0] ):
                    for s in range( 3 ):
                        delete_batch_norm_unet( self.upsampler_nets[l][s][c] )

    
    def forward( self, x ):
        if ( x.shape[1] != self.params_size ):
            raise RuntimeError("distribution_params has channel size {}, but should be {}"
                            .format( x.shape[1], self.params_size ) )
        if ( self.add_ones ):
            x = torch.cat( [ torch.ones( [ x.shape[0], 1, x.shape[2], x.shape[3] ], device = x.device ), x ], dim = 1 )
        return ParallelCNNDistribution( self.output_distribution, self.event_shape, self.base_nets, self.upsampler_nets, x )

    def params_size( self, channels ):
        return self.params_size
