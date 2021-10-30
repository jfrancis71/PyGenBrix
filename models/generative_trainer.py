import argparse
import torchvision
import pytorch_lightning as pl

import PyGenBrix.dist_layers.pixelcnn as pixel_cnn
import PyGenBrix.dist_layers.parallelcnn as parallel_cnn
import PyGenBrix.dist_layers.vdvae as vdvae
import PyGenBrix.Train as Train
import PyGenBrix.dist_layers.common_layers as dl
import PyGenBrix.dist_layers.spatial_independent as sp

ap = argparse.ArgumentParser(description="generative_trainer")
ap.add_argument("--tensorboard_log")
ap.add_argument("--max_epochs", default=10, type=int)
ap.add_argument("--lr", default=.0002, type=float)
ap.add_argument("--batch_size", default=8, type=int)
ap.add_argument("--accumulate_grad_batches", default=8, type=int)
ap.add_argument("--fast_dev_run", action="store_true")
ap.add_argument("--dataset")
ap.add_argument("--rv_distribution")
ap.add_argument("--model")
ap.add_argument("--parallelcnn_num_upsampling_stages", default=1, type=int)
ap.add_argument("--nr_resnet", default=5, type=int)
ap.add_argument("--train_log_freq", default=0, type=int)
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
elif ns.dataset == "celeba64":
    dataset = torchvision.datasets.ImageFolder(root="/home/julian/ImageDataSets/celeba/",
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Pad((-15, -40,-15-1, -30-1)),
            torchvision.transforms.Resize(64), torchvision.transforms.ToTensor(),
        ]))
    image_channels = 3
    image_size = 64
elif ns.dataset == "mnist32":
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
elif ns.rv_distribution == "q3":
    rv_distribution = dl.IndependentQuantizedLayer( num_buckets = 8)
elif ns.rv_distribution == "spiq3":
    rv_distribution = sp.SpatialIndependentDistributionLayer( [image_channels, image_size, image_size], dl.IndependentQuantizedLayer( num_buckets = 8), num_params=30 )
elif ns.rv_distribution == "PixelCNNDiscMixDistribution":
    rv_distribution = pixel_cnn.PixelCNNDiscreteMixLayer()
else:
    print("rv distribution not recognized")
    quit()

if ns.model == "PixelCNN":
    model = pixel_cnn.PixelCNNDistribution([image_channels, image_size, image_size], rv_distribution, nr_resnet=ns.nr_resnet )
elif ns.model == "ParallelCNN":
    model = parallel_cnn.ParallelCNNDistribution([image_channels, image_size, image_size], rv_distribution, max_unet_layers=3, num_upsampling_stages=int(ns.parallelcnn_num_upsampling_stages))
    if ns.rv_distribution == "spiq3":
        print("parallelcnn and spiq3 incompatible")
        quit()
elif ns.model == "VDVAE":
    model = vdvae.VDVAE([image_channels, image_size, image_size], rv_distribution)
else:
    print("model not recognized")
    quit()

trainer = Train.LightningDistributionTrainer(model, dataset, learning_rate=ns.lr, batch_size=ns.batch_size)
train_callback = Train.LogSamplesTrainingCallback(every_global_step=ns.train_log_freq, temperature=1.0) if ns.train_log_freq != 0 else None
pl.Trainer(fast_dev_run=ns.fast_dev_run, gpus=1, accumulate_grad_batches=ns.accumulate_grad_batches, max_epochs=ns.max_epochs, 
           callbacks=[Train.LogSamplesEpochCallback(temperature=1.0), train_callback],
           default_root_dir=ns.tensorboard_log).fit(trainer)
