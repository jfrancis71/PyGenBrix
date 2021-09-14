import argparse
import torchvision
import pytorch_lightning as pl

import PyGenBrix.dist_layers.pixelcnn as cnn
import PyGenBrix.Train as Train
import PyGenBrix.dist_layers.common_layers as dl
import PyGenBrix.dist_layers.spatial_independent as sp

ap = argparse.ArgumentParser(description="LBAE")
ap.add_argument("--tensorboard_log")
ap.add_argument("--nr_resnet", default=5, type=int)
ns = ap.parse_args()


mymodel = cnn.PixelCNNDistribution([ 3, 32, 32 ], sp.SpatialIndependentDistributionLayer( [3, 32, 32], dl.IndependentQuantizedLayer( num_buckets = 8), num_params=30 ), nr_resnet=ns.nr_resnet )


celeba_dataset = torchvision.datasets.ImageFolder(
    root="/home/julian/ImageDataSets/celeba/",
    transform = torchvision.transforms.Compose( [
        torchvision.transforms.Pad( ( -15, -40,-15-1, -30-1) ), torchvision.transforms.Resize( 32 ), torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: dl.quantize(x,8)) ] ))

trainer = Train.LightningDistributionTrainer( mymodel, celeba_dataset, learning_rate = .0002, batch_size = 8 )

pl.Trainer( fast_dev_run = False, gpus=1, accumulate_grad_batches = 8,
          callbacks=[
                     Train.LogDistributionSamplesPerEpoch(temperature=1.0),
                     Train.LogDistributionSamplesPerEpoch(temperature=0.7),
                     Train.LogDistributionModePerEpoch(),
          ], default_root_dir=ns.tensorboard_log).fit( trainer )
