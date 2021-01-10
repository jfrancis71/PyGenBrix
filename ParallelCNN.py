import torch.nn as nn
import torch
import numpy as np
import pl_bolts.models.vision.unet as plt_unet

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

        logging_dict = {}
        output_log_prob = 0.0
        allowed_information = 0.0 * samples
        allowed_information[:,:,::2,::2] += samples[:,:,::2,::2]
        no_channels = len( self.parallelcnns )

        #predict all odd pixels
        block_log_prob = 0.0
        for channel in range( no_channels ):
            network_input = torch.cat( ( allowed_information, self.logits ), dim = 1 )
            network_output_logits = self.parallelcnns[ channel ][ 0 ]( network_input )
            block_log_prob += self.output_distribution( network_output_logits[:,:,1::2,1::2] ).log_prob( samples[:,channel:channel+1,1::2,1::2] )["log_prob"]
            allowed_information[:,channel,1::2,1::2] += samples[:,channel,1::2,1::2]
        logging_dict["block1_log_prob"] = block_log_prob
        output_log_prob += block_log_prob

        #predict all pixels even row, odd column
        block_log_prob = 0.0
        for channel in range( no_channels ):
            network_input = torch.cat( ( allowed_information, self.logits ), dim = 1 )
            network_output_logits = self.parallelcnns[ channel ][ 1 ]( network_input )
            block_log_prob += self.output_distribution( network_output_logits[:,:,::2,1::2] ).log_prob( samples[:,channel:channel+1,::2,1::2] )["log_prob"]
            allowed_information[:,channel,::2,1::2] += samples[:,channel,::2,1::2]
        logging_dict["block2_log_prob"] = block_log_prob
        output_log_prob += block_log_prob

        #predict all pixels odd row, even column
        block_log_prob = 0.0
        for channel in range( no_channels ):
            network_input = torch.cat( ( allowed_information, self.logits ), dim = 1 )
            network_output_logits = self.parallelcnns[ channel ][ 2 ]( network_input )
            block_log_prob += self.output_distribution( network_output_logits[:,:,1::2,::2] ).log_prob( samples[:,channel:channel+1,1::2,::2] )["log_prob"]
            allowed_information[:,channel,1::2,::2] += samples[:,channel,1::2,::2]
        logging_dict["block3_log_prob"] = block_log_prob
        output_log_prob += block_log_prob

        logging_dict["log_prob"] = output_log_prob

        return logging_dict

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

        logging_dict = {}
        
        output_log_prob = 0.0
        allowed_information = 0.0 * bottom_samples
        no_channels = len( self.bottom_parallelcnns )
        #predict all even pixels
        for channel in range( no_channels ):
            network_input = torch.cat( ( allowed_information, bottom_logit_inputs ), dim = 1 )
            network_output_logits = self.bottom_parallelcnns[ channel ]( network_input )
            channel_log_prob = self.output_distribution( network_output_logits ).log_prob( bottom_samples[:,channel:channel+1] )["log_prob"]
            logging_dict["base_log_prob/channel_"+str(channel)] = channel_log_prob
            output_log_prob += channel_log_prob
            allowed_information[:,channel] = bottom_samples[:,channel]
        logging_dict["base_log_prob"] = output_log_prob.clone()
            
        for level in range( self.levels ):
            upsample_log_prob_dict = UpsamplerDistribution(
                self.output_distribution,
                samples[:,:,::2**(self.levels-level),::2**(self.levels-level)],
                self.logits[:,:,::2**(self.levels-level-1),::2**(self.levels-level-1)],
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
        num_logits = output_distribution.params_size( 1 )
        self.bottom_net = nn.ModuleList( [ 
        #Bug in PLT.UNet for num_layers = 0
            ( plt_unet.UNet( num_classes = num_logits, input_channels = 1 + dims[0], num_layers = num_layers ) if num_layers > 0 else default_parallel_cnn_fn( dims, num_logits ) ) for c in range( dims[0] ) ] )
        upsampler_nets = []
        for l in range( upsampling_stages ):
            output_width = bottom_width * 2**(l+1)
            num_layers = int( min( output_width - 1, 3 ) )
            upsampler_nets.append(
                nn.ModuleList( [
                    nn.ModuleList( [ plt_unet.UNet( num_logits, input_channels = 1 + dims[0], num_layers = num_layers ) for s in range(3) ] ) for c in range(dims[0]) ] ) )
        self.upsampling_stages = upsampling_stages
        self.output_distribution = output_distribution
        self.dims = dims
        self.upsampler_nets = nn.ModuleList( upsampler_nets )
    
    def forward( self, logits ):
        return MultiStageParallelCNNDistribution( self.output_distribution, self.dims, self.bottom_net, self.upsampler_nets, self.upsampling_stages, logits )

    def params_size( self, channels ):
        return 1
