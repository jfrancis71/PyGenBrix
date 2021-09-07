#Attempt to do layer by layer training
#didn't really work well for CelebA, by layer 4 learning very little.
#Committed here for record purposes.


import torch
import torch.nn as nn
import torch.nn.functional as F
import PyGenBrix.dist_layers.common_layers as dl
import PyGenBrix.dist_layers.spatial_independent as sp


class AutoencoderBlock(nn.Module):
    def __init__(self, input_shape, out_channels, decoder_distribution):
        super(AutoencoderBlock, self).__init__()
        in_channels = input_shape[0]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1, stride=2, padding=0),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_channels*2, out_channels*2, 1), nn.ReLU(),
            nn.Conv2d(out_channels*2, decoder_distribution.params_size(in_channels), 1))
        self.decoder_layer = decoder_distribution
    
    def log_prob(self, x):
        h = self.encode(x, grad=True)
        xparams = self.decoder(h)
        log_prob = self.decoder_layer(xparams).log_prob(x)
        return log_prob
    
    def encode(self, x, grad=False):
        """encode(self, x, grad) where grads controls if gradient is attached using straight through gradient method"""
        latent_code = self.encoder(x-0.5)
        bin_latent_code = (latent_code>0.0)*1.0
        probs = F.sigmoid(latent_code)
        stg_latent_code = bin_latent_code + probs - probs.detach()
        return stg_latent_code if grad else bin_latent_code

    def decode(self, h):
        xparams = self.decoder(h)
        sample_x = self.decoder_layer(xparams).mode()
        return sample_x


class BinaryAutoencoderBlock(AutoencoderBlock):
    """BinaryAutoencoder autoencodes an input into a latent representation with twice the number of channels
       and half the spatial resolution.
    """
    def __init__(self, input_shape):
        super(BinaryAutoencoderBlock, self).__init__(input_shape, input_shape[0]*2, dl.IndependentBernoulliLayer())

    def log_prob(self, x):
        log_prob = super().log_prob(x)["log_prob"]
        h = self.encode(x, grad=False)
        xr = self.decode(h)
        bit_error = torch.mean((x-xr).abs(), axis=[1,2,3])
        return {"log_prob": log_prob, "bit_error": bit_error}


class BinaryAutoregressiveAutoencoderBlock(AutoencoderBlock):
    """BinaryAutoencoder autoencodes an input into a latent representation with twice the number of channels
       and half the spatial resolution.
    """
    def __init__(self, input_shape):
        super(BinaryAutoregressiveAutoencoderBlock, self).__init__(input_shape, input_shape[0]*2, sp.SpatialIndependentDistributionLayer(input_shape, dl.IndependentBernoulliLayer(), 128))


class QuantizedAutoencoderBlock(AutoencoderBlock):
    def __init__(self, input_shape):
        super(QuantizedAutoencoderBlock, self).__init__(input_shape, 32, dl.IndependentQuantizedLayer(num_buckets=8))


class AutoencoderChain(nn.Module):
    def __init__(self, chain, block):
        super(AutoencoderChain, self).__init__()
        self.chain = chain
        self.block = block
        
    def log_prob(self, x):
        if self.chain is None:
            enc = x
        else:
            enc = self.chain.encode(x).detach()
        return self.block.log_prob(enc)
        
    def sample(self, x, temperature=1.0):
        h = self.encode(x)
        xr = self.decode(h)
        return xr
    
    def mode(self, x):
        return self.sample(x)
    
    def encode(self, x):
        if self.chain is None:
            h = x
        else:
            h = self.chain.encode(x).detach()
        return self.block.encode(h, grad=False)
    
    def decode(self, h):
        h = self.block.decode(h)
        if self.chain is None:
            dec = h
        else:
            dec = self.chain.decode(h)
        return dec
