import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

import PyGenBrix.dist_layers.parallelcnn as cnn
import PyGenBrix.Train as Train
import PyGenBrix.dist_layers.common_layers as dl
import PyGenBrix.dist_layers.spatial_independent as sp
import PyGenBrix.dist_layers.bitwise_factorize as fq

class PixelCNN_Bitwise_Distribution(nn.Module):
    def __init__(self):
        super(PixelCNN_Bitwise_Distribution, self).__init__()
        self.cnn1 = cnn.ParallelCNNDistribution([3, 32, 32], dl.IndependentBernoulliLayer(), num_upsampling_stages = 5, max_unet_layers=3)
        self.cnn2 = cnn.ParallelCNNLayer([3, 32, 32], dl.IndependentBernoulliLayer(), num_upsampling_stages = 5, max_unet_layers=3, num_conditional_channels=3)
        self.cnn3 = cnn.ParallelCNNLayer([3, 32, 32], dl.IndependentBernoulliLayer(), num_upsampling_stages = 5, max_unet_layers=3, num_conditional_channels=3)
        
    def log_prob(self, x):
        bins = fq.dec2bin((x*8).type(torch.uint8), bits=3)
        dist1 = self.cnn1
        q1 = fq.bin2dec(bins[:,:,:,:,:1],bits=1)/2+ 1.0/(2*2.0)
        
        dist2 = self.cnn2(q1-.5)
        q2 = fq.bin2dec(bins[:,:,:,:,:2],bits=2)/4+ 1.0/(4*2.0)

        dist3 = self.cnn3(q2-.5)
        q3 = fq.bin2dec(bins[:,:,:,:,:3],bits=3)/8+ 1.0/(8*2.0)

        l1 = dist1.log_prob(bins[:,:,:,:,0].type(torch.float))["log_prob"]
        l2 = dist2.log_prob(bins[:,:,:,:,1].type(torch.float))["log_prob"]
        l3 = dist3.log_prob(bins[:,:,:,:,2].type(torch.float))["log_prob"]
        lp = l1+l2+l3
        dict = {
            "l1": l1,
            "l2": l2,
            "l3": l3,
            "log_prob": lp
        }
        return dict
    
    def sample(self, temperature=1.0):
        bins = torch.zeros(1, 3, 32, 32, 3).to("cuda")
        dist1 = self.cnn1
        bins[:,:,:,:,0] = dist1.sample(temperature)
        q1 = fq.bin2dec(bins[:,:,:,:,:1],bits=1)/2+ 1.0/(2*2.0)
        
        dist2 = self.cnn2(q1-.5)
        bins[:,:,:,:,1] = dist2.sample(temperature)
        q2 = fq.bin2dec(bins[:,:,:,:,:2],bits=2)/4+ 1.0/(4*2.0)

        dist3 = self.cnn3(q2-.5)
        bins[:,:,:,:,2] = dist3.sample(temperature)
        q3 = fq.bin2dec(bins[:,:,:,:,:3],bits=3)/8+ 1.0/(8*2.0)
        
        decs = q3
        return decs.detach()
    
    def mode(self):
        bins = torch.zeros(1, 3, 32, 32, 3).to("cuda")
        dist1 = self.cnn1
        bins[:,:,:,:,0] = dist1.mode()
        q1 = fq.bin2dec(bins[:,:,:,:,:1],bits=1)/2+ 1.0/(2*2.0)
        
        dist2 = self.cnn2(q1-.5)
        bins[:,:,:,:,1] = dist2.mode()
        q2 = fq.bin2dec(bins[:,:,:,:,:2],bits=2)/4+ 1.0/(4*2.0)

        dist3 = self.cnn3(q2-.5)
        bins[:,:,:,:,2] = dist3.mode()
        q3 = fq.bin2dec(bins[:,:,:,:,:3],bits=3)/8+ 1.0/(8*2.0)
        
        decs = q3
        return decs.detach()

mymodel = PixelCNN_Bitwise_Distribution()

celeba_dataset = torchvision.datasets.ImageFolder(
    root="/home/julian/ImageDataSets/celeba/",
    transform = torchvision.transforms.Compose( [
        torchvision.transforms.Pad( ( -15, -40,-15-1, -30-1) ), torchvision.transforms.Resize( 32 ), torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: dl.quantize(x,8)) ] ))

trainer = Train.LightningDistributionTrainer( mymodel, celeba_dataset, learning_rate = .001, batch_size = 16 )

pl.Trainer( fast_dev_run = False, gpus=1, accumulate_grad_batches = 4,
          callbacks=[
                     Train.LogDistributionSamplesPerEpoch(temperature=1.0),
                     Train.LogDistributionSamplesPerEpoch(temperature=0.7),
                     Train.LogDistributionModePerEpoch(),
          ]).fit( trainer )
