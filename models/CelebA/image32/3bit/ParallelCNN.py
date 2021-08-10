import argparse
import torchvision
import pytorch_lightning as pl

import PyGenBrix.dist_layers.parallelcnn as cnn
import PyGenBrix.Train as Train
import PyGenBrix.dist_layers.common_layers as dl

ap = argparse.ArgumentParser(description="ParallelCNN")
ap.add_argument("--num_upsampling_stages", default=1)
ns = ap.parse_args()

mymodel = cnn.ParallelCNNDistribution([ 3, 32, 32 ], dl.IndependentQuantizedLayer( num_buckets = 8),max_unet_layers=3, num_upsampling_stages=int(ns.num_upsampling_stages))

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
