import math

import numpy as np
import torch
import torch.nn as nn
import pl_bolts.models.vision.unet as plt_unet


base_slice = (slice(0, None, 2), slice(0, None, 2))
upsampling_slices = [(slice(1, None, 2), slice(1, None, 2)),
    (slice(0, None, 2), slice(1, None, 2)),
    (slice( 1, None, 2), slice(0, None, 2))]


def delete_batch_norm_unet(unet):
    del unet.layers[0].net[1]
    del unet.layers[0].net[3]
    for l in range(1, len(unet.layers)//2):
        del unet.layers[l].net[1].net[1]
        del unet.layers[l].net[1].net[3]
    for l in range((len( unet.layers)+1)//2, len(unet.layers)-1 ):
        del unet.layers[l].conv.net[1]
        del unet.layers[l].conv.net[3]       


class _ParallelCNNDistribution(nn.Module):
    def __init__(self, event_shape, output_distribution_layer, num_upsampling_stages, max_unet_layers=3, num_conditional_channels=None):
        super(_ParallelCNNDistribution, self).__init__()
        self.output_distribution_layer = output_distribution_layer
        num_output_distribution_params = output_distribution_layer.params_size(1)
        input_channels = event_shape[0]
        if num_conditional_channels is not None:
            input_channels += num_conditional_channels
        base_width = event_shape[1]/2**num_upsampling_stages
        unet_num_layers = int(min( math.log( base_width )+1, max_unet_layers))
        self.base_nets = nn.ModuleList([
            plt_unet.UNet(num_classes=num_output_distribution_params, input_channels=input_channels, num_layers=unet_num_layers) for c in range(event_shape[0])])
        upsampler_nets = []
        for l in range(num_upsampling_stages):
            output_width = base_width * 2**(l+1)
            unet_num_layers = int(min(math.log(output_width)+1, max_unet_layers))
            upsampler_nets.append(
                nn.ModuleList([
                    nn.ModuleList([plt_unet.UNet(num_output_distribution_params, input_channels=input_channels, num_layers=unet_num_layers) for c in range(event_shape[0])]) for s in range(3)]))
        self.num_upsampling_stages = len(upsampler_nets)
        self.upsampler_nets = upsampler_nets
        self.event_shape = event_shape
        self.num_conditional_channels = num_conditional_channels

#Note this will alter masked_value
    def log_prob_block(self, upsampled_images, distribution_params, masked_value, block_parallel_cnns, slice):
        block_log_prob = 0.0
        for channel in range(len(block_parallel_cnns)):
            if distribution_params is not None:
                network_input = torch.cat((masked_value, distribution_params), dim=1 )
            else:
                network_input = torch.cat([masked_value], dim=1)
            output_distribution_params = block_parallel_cnns[channel](network_input)
            block_log_prob += self.output_distribution_layer(output_distribution_params[:,:,slice[0],slice[1]]).log_prob(upsampled_images[:,channel:channel+1,slice[0],slice[1]])["log_prob"]
            masked_value[:,channel,slice[0],slice[1]] = upsampled_images[:,channel,slice[0],slice[1]]
        return block_log_prob

#Note, alters samples
    def sample_block(self, samples, distribution_params, block_parallel_cnns, slice):
        for channel in range(len(block_parallel_cnns)):
            network_input = torch.cat((samples, distribution_params), dim=1)
            output_distribution_params = block_parallel_cnns[channel](network_input)
            samples[:,channel,slice[0],slice[1]] = self.output_distribution_layer(output_distribution_params).sample()[:,0,slice[0],slice[1]]

    def upsampler_log_prob(self, value, distribution_params, parallel_cnns):
        logging_dict = {}
        masked_value = torch.zeros_like(value)
        masked_value[:,:,base_slice[0],base_slice[1]] = value[:,:,base_slice[0],base_slice[1]]
        log_prob = 0.0
        for s in range(3):
            block_log_prob = self.log_prob_block(value, distribution_params, masked_value, parallel_cnns[s], upsampling_slices[s])
            logging_dict["block"+str(s)+"_log_prob"] = block_log_prob
            log_prob += block_log_prob
        logging_dict["log_prob"] = log_prob
        return logging_dict

    def upsampler_sample(self, downsampled_images, distribution_params, parallel_nets):
        samples = torch.zeros([downsampled_images.shape[0], downsampled_images.shape[1], downsampled_images.shape[2]*2, downsampled_images.shape[3]*2], device=downsampled_images.device)
        samples[:,:,base_slice[0],base_slice[1]] = downsampled_images
        for s in range(3):
            self.sample_block(samples, distribution_params, parallel_nets[s], upsampling_slices[s])
        return samples

    def log_prob(self, value, conditionals=None):
        if self.num_conditional_channels is not None:
            if value.size()[0] != conditionals.size()[0]:
                raise RuntimeError("value batch size {}, but conditionals has batch size {}"
                            .format(value.shape[0], conditionals.shape[0]))
        if value.size()[1:4] != torch.Size( self.event_shape):
            raise RuntimeError("value shape  {}, but event_shape has shape {}"
                            .format(value.shape[1:4], self.event_shape))
        base_samples = value[:,:,::2**self.num_upsampling_stages,::2**self.num_upsampling_stages]
        if conditionals is not None:
            base_distribution_params = conditionals[:,:,::2**self.num_upsampling_stages,::2**self.num_upsampling_stages]
        else:
            base_distribution_params = None
        logging_dict = {}
        log_prob = 0.0
        masked_value = torch.zeros_like(base_samples)

        #predict all base pixels
        log_prob = self.log_prob_block(base_samples, base_distribution_params, masked_value, self.base_nets, base_slice)
        logging_dict["base_log_prob"] = log_prob.clone()
        for level in range( self.num_upsampling_stages ):
            level_subsample_rate = 2**(self.num_upsampling_stages-level-1)
            if conditionals is not None:
                upsample_distribution_params = conditionals[:,:,::level_subsample_rate,::level_subsample_rate]
            else:
               upsample_distribution_params = None
            upsample_log_prob_dict = self.upsampler_log_prob(
                value[:,:,::level_subsample_rate,::level_subsample_rate],
                upsample_distribution_params,
                self.upsampler_nets[ level ] )
            for k, v in upsample_log_prob_dict.items():
                logging_dict["upsample_level_"+str(level)+"/"+k] = upsample_log_prob_dict[k]
            log_prob += upsample_log_prob_dict["log_prob"]
        logging_dict["log_prob"] = log_prob
        return logging_dict
    
    def sample(self):
        with torch.no_grad():
            sample = torch.zeros([self.distribution_params.shape[0], self.event_shape[0], self.event_shape[1]//2**self.num_upsampling_stages, self.event_shape[2]//2**self.num_upsampling_stages ], device=self.distribution_params.device)
            base_distribution_params = self.distribution_params[:,:,::2**self.num_upsampling_stages,::2**self.num_upsampling_stages]
            self.sample_block(sample, base_distribution_params, self.base_parallel_nets, base_slice)
            for level in range(self.num_upsampling_stages):
                level_subsample_rate = 2**(self.num_upsampling_stages-level-1)
                sample = self.upsampler_sample(
                    sample,
                    self.distribution_params[:,:,::level_subsample_rate,::level_subsample_rate],
                    self.upsample_parallel_nets[ level ])
        return sample

class ParallelCNNDistribution(nn.Module):
    def __init__(self, event_shape, output_distribution_layer, num_upsampling_stages, max_unet_layers=3):
        super(ParallelCNNDistribution, self).__init__()
        self.distribution = ParallelCNNDistribution(event_shape, output_distribution_layer, num_upsampling_stages, max_unet_layers=3)

    def log_prob(self, samples):
        return self.distribution.log_prob(samples)

    def sample(self):
        return self.distribution.sample()

class _ParallelCNNLayerDistribution(nn.Module):
    def __init__(self, distribution, params):
        super(_ParallelCNNLayerDistribution, self).__init__()
        self.distribution = distribution
        self.params = params

    def log_prob(self, samples):
        return self.distribution.log_prob(samples, self.params)

    def sample(self):
        with torch.no_grad():
            return self.distribution.sample(self.params)


class ParallelCNNLayer(nn.Module):
    def __init__(self, event_shape, output_distribution_layer, num_upsampling_stages, max_unet_layers, num_conditional_channels):
        super(ParallelCNNLayer, self).__init__()
        self.distribution = _ParallelCNNDistribution(event_shape, output_distribution_layer, num_upsampling_stages, max_unet_layers=3, num_conditional_channels=num_conditional_channels)

    def forward(self, x):
        return _ParallelCNNLayerDistribution(self.distribution, x)
