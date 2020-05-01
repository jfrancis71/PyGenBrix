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
    information_masks = np.array( [ np.sum( pixel_channel_groups[:x], axis=0 ) if x > 0 else np.zeros( [ dims[0], dims[1], dims[2] ] ) for x in range(4*dims[2]) ] )
    return information_masks

def create_parallelcnns( dims ):
    return [ torch.nn.Sequential(
#        torch.nn.Conv2d( 2,1,3, padding=1 ), nn.Tanh(),
#        torch.nn.Conv2d( 1, 1, 1), nn.Tanh(),
        torch.nn.Conv2d( 2, dims[0]*1, 3, padding=1 )
        
) for x in range(4*dims[0]) ]

class ParallelCNN( nn.Module ):

    def __init__( self, dims ):
        super(ParallelCNN, self).__init__()
        self.parallelcnns = create_parallelcnns( dims )
        self.pixel_channel_groups = generate_pixel_channel_groups( dims )
        self.information_masks = generate_information_masks( dims )
        self.position_layer = torch.nn.Parameter( torch.tensor( np.zeros( [ 1, 28, 28 ] ).astype( np.float32 )  ), requires_grad=True )
        
    def log_prob( self, samples, conditional_input = None ):

        i0 = torch.cat( ( samples*torch.tensor( self.information_masks[0].astype( np.float32 ) ), self.position_layer.expand_as( samples ) ), dim=1 )
        log0 = self.parallelcnns[0]( i0 )
#        log0 = i0[:,1:2]
        l0 = torch.distributions.Bernoulli( logits = log0 ).log_prob( samples )
        t0 = torch.tensor( self.pixel_channel_groups[0] ) *l0

        i1 = torch.cat( ( samples*torch.tensor( self.information_masks[1].astype( np.float32 ) ), self.position_layer.expand_as( samples ) ), dim=1 )
        log1 = self.parallelcnns[1]( i1 )
        l1 = torch.distributions.Bernoulli( logits = log1 ).log_prob( samples )
        t1 = torch.tensor( self.pixel_channel_groups[1] ) *l1

        i2 = torch.cat( ( samples*torch.tensor( self.information_masks[2].astype( np.float32 ) ), self.position_layer.expand_as( samples ) ), dim=1 )
        log2 = self.parallelcnns[2]( i2 )
        l2 = torch.distributions.Bernoulli( logits = log2 ).log_prob( samples )
        t2 = torch.tensor( self.pixel_channel_groups[2] ) *l2
        
        i3 = torch.cat( ( samples*torch.tensor( self.information_masks[3].astype( np.float32 ) ), self.position_layer.expand_as( samples ) ), dim=1 )
        log3 = self.parallelcnns[3]( i3 )
        l3 = torch.distributions.Bernoulli( logits = log3 ).log_prob( samples )
        t3 = torch.tensor( self.pixel_channel_groups[3] ) *l3

        return torch.sum( t0 + t1 + t2 + t3 )
    
    def sample( self, conditional_input = None ):
        sample = torch.tensor( np.zeros( [ 1, 1, 28, 28 ] ).astype( np.float32 ) )
        
        i0 = torch.cat(
            ( sample*torch.tensor( self.information_masks[0].astype( np.float32 ) ),
            self.position_layer.expand_as( sample ) ), dim=1 )
        log0 = self.parallelcnns[0]( i0 )
        l0 = torch.distributions.Bernoulli( logits = log0 ).sample()
        t0 = torch.tensor( self.pixel_channel_groups[0] ) *l0
        sample += t0

        i1 = torch.cat(
            ( sample*torch.tensor( self.information_masks[1].astype( np.float32 ) ),
            self.position_layer.expand_as( sample ) ), dim=1 )
        log1 = self.parallelcnns[1]( i1 )
        l1 = torch.distributions.Bernoulli( logits = log1 ).sample()
        t1 = torch.tensor( self.pixel_channel_groups[1] ) *l1
        sample += t1

        i2 = torch.cat(
            ( sample*torch.tensor( self.information_masks[2].astype( np.float32 ) ),
            self.position_layer.expand_as( sample ) ), dim=1 )
        log2 = self.parallelcnns[2]( i2 )
        l2 = torch.distributions.Bernoulli( logits = log2 ).sample()
        t2 = torch.tensor( self.pixel_channel_groups[2] ) *l2
        sample += t2

        i3 = torch.cat(
            ( sample*torch.tensor( self.information_masks[3].astype( np.float32 ) ),
            self.position_layer.expand_as( sample ) ), dim=1 )
        log3 = self.parallelcnns[3]( i3 )
        l3 = torch.distributions.Bernoulli( logits = log3 ).sample()
        t3 = torch.tensor( self.pixel_channel_groups[3] ) *l3
        sample += t3

        return sample
