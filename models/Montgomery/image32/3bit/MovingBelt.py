import itertools

import torch
import torchvision

from matplotlib import pyplot as plt
import pytorch_lightning as pl

from PyGenBrix import DistributionLayers as dl
from PyGenBrix import PixelCNNDistribution as pcnn
from PyGenBrix import SpatialIndependentDistribution as sp
from PyGenBrix import Train

import LBAE.models5
import LBAE.parameters

train_dataset = torchvision.datasets.ImageFolder(
    root="/home/julian/PyGenBrixProj/Montgomery/MovingBelt/training_images",
    transform = torchvision.transforms.Compose( [
        torchvision.transforms.Resize( 32 ),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: dl.quantize(x,8)) ] ) )

valid_dataset = torchvision.datasets.ImageFolder(
    root="/home/julian/PyGenBrixProj/Montgomery/MovingBelt/validation_images",
    transform = torchvision.transforms.Compose( [
        torchvision.transforms.Resize( 32 ),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: dl.quantize(x,8)) ] ) )

class FixedSequenceSizeRecurrentDataSet( torch.utils.data.Dataset ):
    def __init__( self, dataset, size ):
        super( FixedSequenceSizeRecurrentDataSet, self ).__init__()
        self.start = torch.randint(0,len(dataset)-5,(size,))
        self.len = torch.ones(size, dtype=torch.int) * 4
        self.dataset = dataset
        self.size = size
        
    def __len__( self ):
        return self.size
    
    def __getitem__( self, idx ):
        return (torch.stack([ self.dataset[self.start[idx]+i][0] for i in range(self.len[idx]) ], dim=0),0)

train_recurrent_dataset = FixedSequenceSizeRecurrentDataSet(train_dataset, 20000)
valid_recurrent_dataset = FixedSequenceSizeRecurrentDataSet(valid_dataset, 2000)

hps = LBAE.parameters.Params()
hps.channels = 3
hps.img_size = 32
hps.vae = False
hps.zsize = 128
hps.zround = -1
hps.dataset = 'celeba'
hps.channels_out = 64

class SequenceDistribution(torch.nn.Module):
    def __init__(self):
        super(SequenceDistribution, self).__init__()
        self.image_enc = LBAE.models5.EncConvResBlock32(hps)
        self.image_dist_layer = pcnn.PixelCNNLayer([ 3, 32, 32 ], num_conditional=128, output_distribution_layer=sp.SpatialIndependentDistributionLayer( [3, 32, 32],
            dl.IndependentQuantizedLayer( num_buckets = 8), num_params=30 ) )
        self.n1 = torch.nn.Sequential(torch.nn.Linear(256,256),torch.nn.Tanh(),torch.nn.Linear(256,128))

    def log_prob(self, sequence):
        h0 = torch.zeros((sequence.shape[0], 128)).cuda()
        enc1 = self.image_enc(sequence[:,0])[0]
        h1 = self.n1(torch.cat([h0,enc1], dim=1))
        log1 = self.image_dist_layer(h1).log_prob(sequence[:,1])
        enc2 = self.image_enc(sequence[:,1])[0]
        h2 = self.n1(torch.cat([h1,enc2], dim=1))
        log2 = self.image_dist_layer(h2).log_prob(sequence[:,2])        
        enc3 = self.image_enc(sequence[:,2])[0]
        h3 = self.n1(torch.cat([h1,enc3], dim=1))
        log3 = self.image_dist_layer(h3).log_prob(sequence[:,3])
        return {"log_prob": log1["log_prob"]+log2["log_prob"]+log3["log_prob"]}
        
    def sample(self, start):
        self.cuda()
        h0 = torch.zeros((start.shape[0], 128)).cuda()
        enc1 = self.image_enc(start)[0]
        h1 = self.n1(torch.cat([h0,enc1], dim=1))
        sample_image1 = self.image_dist_layer(h1).sample()
        enc2 = self.image_enc(sample_image1)[0]
        h2 = self.n1(torch.cat([h1,enc2], dim=1))
        sample_image2 = self.image_dist_layer(h2).sample()
        enc3 = self.image_enc(sample_image2)[0]
        h3 = self.n1(torch.cat([h2,enc3], dim=1))
        sample_image3 = self.image_dist_layer(h3).sample()
        return torch.stack( [ start, sample_image1, sample_image2, sample_image3 ], dim=1).detach()

myseq = SequenceDistribution()

class SequenceTrainer(Train.LightningDistributionTrainer):
    def __init__(self, model, train_dataset, valid_dataset, learning_rate=.0001, batch_size=1):
        self.seq_train_dataset = train_dataset
        self.seq_valid_dataset = valid_dataset
        super(SequenceTrainer, self).__init__( model, 0, add_graph=False, learning_rate=learning_rate, batch_size=batch_size)
        
    def get_datasets(self):
        return (self.seq_train_dataset, self.seq_valid_dataset)

trainer = SequenceTrainer( myseq, train_recurrent_dataset, valid_recurrent_dataset, learning_rate = .0002, batch_size = 4 )

class SequenceLogDistributionPerEpoch(pl.Callback):
    def __init__(self):
        super(SequenceLogDistributionPerEpoch, self).__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        samples = torch.cat([torch.clip(pl_module.model.sample(torch.unsqueeze(valid_recurrent_dataset[i][0][0],dim=0).cuda()),0.0,1.0)[0] for i in range(8)])
        image = torchvision.utils.make_grid(samples, padding=10, nrow=4 )
        print( "LOGGING EPOCH", pl_module.current_epoch )
        pl_module.logger.experiment.add_image("epoch", image, pl_module.current_epoch, dataformats="CHW")
        pl_module.logger.experiment.add_image("epoch"+str(pl_module.current_epoch), image, pl_module.current_epoch, dataformats="CHW")

pl.Trainer(fast_dev_run = False, gpus=1, accumulate_grad_batches=16, callbacks=[SequenceLogDistributionPerEpoch()]).fit(trainer)
