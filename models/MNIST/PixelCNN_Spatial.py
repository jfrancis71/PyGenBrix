import argparse
import torchvision
import pytorch_lightning as pl

import PyGenBrix.dist_layers.pixelcnn as cnn
import PyGenBrix.Train as Train
import PyGenBrix.dist_layers.common_layers as dl
import PyGenBrix.dist_layers.spatial_independent as sp

ap = argparse.ArgumentParser(description="PixelCNN")
ap.add_argument("--tensorboard_log")
ap.add_argument("--max_epochs", default=10, type=int)
ap.add_argument("--nr_resnet", default=5, type=int)
ap.add_argument("--lr", default=.0002, type=float)
ns = ap.parse_args()


mymodel = cnn.PixelCNNDistribution([ 1, 28, 28 ], dl.IndependentBernoulliLayer(), nr_resnet=ns.nr_resnet )

mnist_dataset = torchvision.datasets.MNIST('/home/julian/ImageDataSets/MNIST',
    train=True, download=False,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: 1.0-((x<0.5)*1.0)) ]))

trainer = Train.LightningDistributionTrainer( mymodel, mnist_dataset, learning_rate = ns.lr, batch_size = 8 )

pl.Trainer( fast_dev_run = False, gpus=1, accumulate_grad_batches = 8, max_epochs=ns.max_epochs, 
          callbacks=[
                     Train.LogSamplesEpochCallback(temperature=1.0)
          ], default_root_dir=ns.tensorboard_log).fit( trainer )
