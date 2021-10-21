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
ap.add_argument("--fast_dev_run", action="store_true")
ap.add_argument("--dataset")
ap.add_argument("--rv_distribution")
ns = ap.parse_args()
if ns.dataset == "cifar10":
    dataset = torchvision.datasets.CIFAR10(root='/home/julian/ImageDataSets/CIFAR10', train=True,
        download=False, transform=torchvision.transforms.ToTensor())
    image_channels = 3
    image_size = 32
elif ns.dataset == "celeba32":
    dataset = torchvision.datasets.ImageFolder(root="/home/julian/ImageDataSets/celeba/",
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Pad((-15, -40,-15-1, -30-1)),
            torchvision.transforms.Resize(32), torchvision.transforms.ToTensor(),
        ]))
    image_channels = 3
    image_size = 32
elif ns.dataset == "mnist":
    dataset = torchvision.datasets.MNIST('/home/julian/ImageDataSets/MNIST',
    train=True, download=False,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: 1.0-((x<0.5)*1.0))]))
    image_channels = 1
    image_size = 32
else:
    print("Dataset not recognized.")
    quit()
if ns.rv_distribution == "bernoulli":
    rv_distribution = dl.IndependentBernoulliLayer()
elif ns.rv_distribution == "spiq3":
    rv_distribution = sp.SpatialIndependentDistributionLayer( [image_channels, 32, 32], dl.IndependentQuantizedLayer( num_buckets = 8), num_params=30 )
elif ns.rv_distribution == "PixelCNNDiscMixDistribution":
    rv_distribution = cnn.PixelCNNDiscreteMixLayer()
else:
    print("rv distribution not recognized")
    quit()
model = cnn.PixelCNNDistribution([ image_channels, image_size, image_size ], rv_distribution, nr_resnet=ns.nr_resnet )
trainer = Train.LightningDistributionTrainer( model, dataset, learning_rate = ns.lr, batch_size = 8 )
pl.Trainer(fast_dev_run=ns.fast_dev_run, gpus=1, accumulate_grad_batches=8, max_epochs=ns.max_epochs, 
          callbacks=[
                     Train.LogSamplesEpochCallback(temperature=1.0)
          ], default_root_dir=ns.tensorboard_log).fit(trainer)
