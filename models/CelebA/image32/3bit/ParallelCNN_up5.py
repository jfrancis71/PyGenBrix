import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import pytorch_lightning as pl

import PyGenBrix.dist_layers.parallelcnn as cnn
import PyGenBrix.Train as Train
import PyGenBrix.dist_layers.common_layers as dl

mymodel = cnn.ParallelCNNDistribution([ 3, 32, 32 ], dl.IndependentQuantizedLayer( num_buckets = 8),max_unet_layers=3, num_upsampling_stages=5 )

celeba_dataset = torchvision.datasets.ImageFolder(
    root="/home/julian/ImageDataSets/celeba/",
    transform = torchvision.transforms.Compose( [
        torchvision.transforms.Pad( ( -15, -40,-15-1, -30-1) ), torchvision.transforms.Resize( 32 ), torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: dl.quantize(x,8)) ] ))

trainer = Train.LightningDistributionTrainer( mymodel, celeba_dataset, learning_rate = .001, batch_size = 16 )

pl.Trainer( fast_dev_run = False, gpus=1, accumulate_grad_batches = 1,
          callbacks=[
                     Train.LogDistributionSamplesPerEpoch(temperature=1.0),
                     Train.LogDistributionSamplesPerEpoch(temperature=0.7),
                     Train.LogDistributionModePerEpoch(),
          ]).fit( trainer )
