import torch
from torch import nn as nn
import torchvision
import itertools

from matplotlib import pyplot as plt

from PyGenBrix import PixelCNNDistribution as cnn
from PyGenBrix import Train as Train
from PyGenBrix import DistributionLayers as dl
from PyGenBrix import SpatialIndependentDistribution as sp

import LBAE.models5
import LBAE.parameters

import pytorch_lightning as pl

hps = LBAE.parameters.Params()
hps.channels = 3
hps.img_size = 32
hps.vae = False
hps.zsize = 128
hps.zround = -1
hps.dataset = 'celeba'
hps.channels_out = 64

class LBDistribution(nn.Module):
    def __init__(self):
        super(LBDistribution, self).__init__()
        self.enc = LBAE.models5.EncConvResBlock32(hps)
        self.cnn = cnn.PixelCNNLayer([ 3, 32, 32 ], num_conditional=128, output_distribution_layer=sp.SpatialIndependentDistributionLayer( [3, 32, 32] , dl.IndependentQuantizedLayer( num_buckets = 8), num_params=30 ) )
        
    def log_prob(self, x):
        enc = self.enc(x)[0]
        dist = self.cnn(enc)
        l = dist.log_prob(x)
        return l

    def sample(self, inp, temperature=1.0):
        enc = self.enc(inp)[0].detach()
        dist = self.cnn(enc)
        return dist.sample(temperature)

    def mode(self, inp):
        enc = self.enc(inp)[0].detach()
        dist = self.cnn(enc)
        return dist.mode()

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

mymodel = LBDistribution()

class TensorboardEpochCallback(pl.Callback):
    def __init__(self):
        super(TensorboardEpochCallback, self).__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        sample_imglist = [ [
            pl_module.val_set[i*2][0].cuda(), 
            pl_module.model.sample( torch.unsqueeze(pl_module.val_set[i*2][0].cuda(),0) )[0],
            pl_module.val_set[i*2+1][0].cuda(), 
            pl_module.model.sample( torch.unsqueeze(pl_module.val_set[i*2+1][0].cuda(),0) )[0] ] for i in range(4) ]
        sample_imglist_t7 = [ [
            pl_module.val_set[i*2][0].cuda(), 
            pl_module.model.sample( torch.unsqueeze(pl_module.val_set[i*2][0].cuda(),0), temperature=0.7 )[0],
            pl_module.val_set[i*2+1][0].cuda(), 
            pl_module.model.sample( torch.unsqueeze(pl_module.val_set[i*2+1][0].cuda(),0), temperature=0.7 )[0] ] for i in range(4) ]
        mode_imglist = [ [
            pl_module.val_set[i*2][0].cuda(), 
            pl_module.model.mode( torch.unsqueeze(pl_module.val_set[i*2][0].cuda(),0) )[0],
            pl_module.val_set[i*2+1][0].cuda(), 
            pl_module.model.mode( torch.unsqueeze(pl_module.val_set[i*2+1][0].cuda(),0) )[0] ] for i in range(4) ]

        trainer.logger.experiment.add_image("samples_T=1.0", torchvision.utils.make_grid(list(itertools.chain(*sample_imglist)), padding=10, nrow=4 ), pl_module.global_step, dataformats="CHW")
        trainer.logger.experiment.add_image("samples_T=0.7", torchvision.utils.make_grid(list(itertools.chain(*sample_imglist_t7)), padding=10, nrow=4 ), pl_module.global_step, dataformats="CHW")
        trainer.logger.experiment.add_image("mode", torchvision.utils.make_grid(list(itertools.chain(*mode_imglist)), padding=10, nrow=4 ), pl_module.global_step, dataformats="CHW")

class SequenceTrainer(Train.LightningDistributionTrainer):
    def __init__(self, model, train_dataset, valid_dataset, learning_rate=.0001,
 batch_size=1):
        self.seq_train_dataset = train_dataset
        self.seq_valid_dataset = valid_dataset
        super(SequenceTrainer, self).__init__( model, 0, add_graph=False, learning_rate=learning_rate, batch_size=batch_size)
        
    def get_datasets(self):
        return (self.seq_train_dataset, self.seq_valid_dataset)

trainer = SequenceTrainer( mymodel, train_dataset, valid_dataset, learning_rate = .0002, batch_size = 8 )

pl.Trainer( fast_dev_run = False, gpus=1, accumulate_grad_batches=8, callbacks=[TensorboardEpochCallback()] ).fit( trainer )
