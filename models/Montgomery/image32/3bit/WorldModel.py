import itertools

import torch
import torchvision
import torch.nn as nn

from matplotlib import pyplot as plt
import pytorch_lightning as pl

from PyGenBrix import DistributionLayers as dl
from PyGenBrix import PixelCNNDistribution as pcnn
from PyGenBrix import SpatialIndependentDistribution as sp
from PyGenBrix import Train
from PyGenBrix import PixelCNNDistribution as cnn

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
        self.start = torch.randint(0,len(dataset)-9,(size,))
        self.len = torch.ones(size, dtype=torch.int) * 8
        self.dataset = dataset
        self.size = size
        
    def __len__( self ):
        return self.size
    
    def __getitem__( self, idx ):
        return (torch.stack([ self.dataset[self.start[idx]+i][0] for i in range(self.len[idx]) ], dim=0),0)

train_recurrent_dataset = FixedSequenceSizeRecurrentDataSet(train_dataset, 10000)
valid_recurrent_dataset = FixedSequenceSizeRecurrentDataSet(valid_dataset, 1000)

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
        self.cnn = cnn.PixelCNNLayer([ 3, 32, 32 ], num_conditional=128, output_distribution_layer=sp.SpatialIndependentDistributionLayer( [3, 32, 32] , dl.IndependentQuantizedLayer( num_buckets = 8), num_params =30 ) )
        
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

lbae_pixelcnn_model = LBDistribution()

class SequenceTrainer(Train.LightningDistributionTrainer):
    def __init__(self, model, train_dataset, valid_dataset, learning_rate=.0001,
 batch_size=1):
        self.seq_train_dataset = train_dataset
        self.seq_valid_dataset = valid_dataset
        super(SequenceTrainer, self).__init__( model, 0, add_graph=False, learning_rate=learning_rate, batch_size=batch_size)
        
    def get_datasets(self):
        return (self.seq_train_dataset, self.seq_valid_dataset)

_ = SequenceTrainer.load_from_checkpoint("~/PyGenBrixProj/Montgomery/MovingBelt/image32/3bit/lightning_logs/PixelCNN_z128/checkpoints/epoch=38-step=12323.ckpt", model=lbae_pixelcnn_model, train_dataset=train_dataset, valid_dataset=valid_dataset)

def tan2bin(x):
    return (1+x)/2
def bin2tan(x):
    return (x*2)-1

class SequenceDistribution(torch.nn.Module):
    def __init__(self):
        super(SequenceDistribution, self).__init__()
        self.n1 = torch.nn.Sequential(torch.nn.Linear(256,256),torch.nn.Tanh(),torch.nn.Linear(256,128))
        self.output = sp.SpatialIndependentDistributionLayer([128, 1, 1] , dl.IndependentBernoulliLayer(), num_params=128, net_size=64)

    def log_prob(self, sequence):
        enc0 = lbae_pixelcnn_model.enc(sequence[:,0])[0].detach()
        enc1 = lbae_pixelcnn_model.enc(sequence[:,1])[0].detach()
        enc2 = lbae_pixelcnn_model.enc(sequence[:,2])[0].detach()
        enc3 = lbae_pixelcnn_model.enc(sequence[:,3])[0].detach()
        enc4 = lbae_pixelcnn_model.enc(sequence[:,4])[0].detach()
        enc5 = lbae_pixelcnn_model.enc(sequence[:,5])[0].detach()
        enc6 = lbae_pixelcnn_model.enc(sequence[:,6])[0].detach()
        enc7 = lbae_pixelcnn_model.enc(sequence[:,7])[0].detach()
        h0 = torch.zeros((sequence.shape[0], 128)).cuda()
        h1 = self.n1(torch.cat([h0,enc0], dim=1))
        log1 = self.output(h1.view((-1,128,1,1))).log_prob(tan2bin(enc1.view((-1,128,1,1))))
        h2 = self.n1(torch.cat([h1,enc1], dim=1))
        log2 = self.output(h2.view((-1,128,1,1))).log_prob(tan2bin(enc2.view((-1,128,1,1))))
        h3 = self.n1(torch.cat([h2,enc2], dim=1))
        log3 = self.output(h3.view((-1,128,1,1))).log_prob(tan2bin(enc3.view((-1,128,1,1))))
        h4 = self.n1(torch.cat([h3,enc3], dim=1))
        log4 = self.output(h4.view((-1,128,1,1))).log_prob(tan2bin(enc4.view((-1,128,1,1))))
        h5 = self.n1(torch.cat([h4,enc4], dim=1))
        log5 = self.output(h5.view((-1,128,1,1))).log_prob(tan2bin(enc5.view((-1,128,1,1))))
        h6 = self.n1(torch.cat([h5,enc5], dim=1))
        log6 = self.output(h6.view((-1,128,1,1))).log_prob(tan2bin(enc6.view((-1,128,1,1))))
        h7 = self.n1(torch.cat([h6,enc6], dim=1))
        log7 = self.output(h7.view((-1,128,1,1))).log_prob(tan2bin(enc7.view((-1,128,1,1))))
        return {"log_prob": log1["log_prob"]+log2["log_prob"]+log3["log_prob"]+log4["log_prob"]+log5["log_prob"]+log6["log_prob"]+log7["log_prob"]}
    
    def sample(self, start):
        self.cuda()
        h0 = torch.zeros((start.shape[0], 128)).cuda()
        enc0 = lbae_pixelcnn_model.enc(start)[0]
        h1 = self.n1(torch.cat([h0,enc0], dim=1))
        enc1 = bin2tan(self.output(h1.view(-1,128,1,1)).sample())[:,:,0,0]
        h2 = self.n1(torch.cat([h1,enc1], dim=1))
        enc2 = bin2tan(self.output(h2.view(-1,128,1,1)).sample())[:,:,0,0]
        h3 = self.n1(torch.cat([h2,enc2], dim=1))
        enc3 = bin2tan(self.output(h3.view(-1,128,1,1)).sample())[:,:,0,0]
        h4 = self.n1(torch.cat([h3,enc3], dim=1))
        enc4 = bin2tan(self.output(h4.view(-1,128,1,1)).sample())[:,:,0,0]
        h5 = self.n1(torch.cat([h4,enc4], dim=1))
        enc5 = bin2tan(self.output(h5.view(-1,128,1,1)).sample())[:,:,0,0]
        h6 = self.n1(torch.cat([h5,enc5], dim=1))
        enc6 = bin2tan(self.output(h6.view(-1,128,1,1)).sample())[:,:,0,0]
        h7 = self.n1(torch.cat([h6,enc6], dim=1))
        enc7 = bin2tan(self.output(h7.view(-1,128,1,1)).sample())[:,:,0,0]
        
        sample_image1 = lbae_pixelcnn_model.cnn(enc1).sample()
        sample_image2 = lbae_pixelcnn_model.cnn(enc2).sample()
        sample_image3 = lbae_pixelcnn_model.cnn(enc3).sample()
        sample_image4 = lbae_pixelcnn_model.cnn(enc4).sample()
        sample_image5 = lbae_pixelcnn_model.cnn(enc5).sample()
        sample_image6 = lbae_pixelcnn_model.cnn(enc6).sample()
        sample_image7 = lbae_pixelcnn_model.cnn(enc7).sample()

        return torch.stack( [ start, sample_image1, sample_image2, sample_image3, sample_image4, sample_image5, sample_image6, sample_image7 ], dim=1).detach()

myseq = SequenceDistribution()

cudarise=myseq.cuda()
cudarise=lbae_pixelcnn_model.cuda()

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
        samples = [torch.clip(pl_module.model.sample(torch.unsqueeze(valid_recurrent_dataset[i][0][0],dim=0).cuda()),0.0,1.0)[0] for i in range(4)]
        image = torchvision.utils.make_grid(list(itertools.chain(*samples)), padding=10, nrow=8 )
        print( "LOGGING EPOCH", pl_module.current_epoch )
        pl_module.logger.experiment.add_image("epoch", image, pl_module.current_epoch, dataformats="CHW")
        pl_module.logger.experiment.add_image("epoch"+str(pl_module.current_epoch), image, pl_module.current_epoch, dataformats="CHW")

pl.Trainer(fast_dev_run = False, gpus=1, accumulate_grad_batches=16, callbacks=[SequenceLogDistributionPerEpoch()]).fit(trainer)
