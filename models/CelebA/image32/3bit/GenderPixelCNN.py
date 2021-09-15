import argparse
import torch
import torchvision
import pytorch_lightning as pl

import PyGenBrix.dist_layers.pixelcnn as pcnn
import PyGenBrix.Train as Train
import PyGenBrix.dist_layers.common_layers as dl
import PyGenBrix.dist_layers.spatial_independent as sp


ap = argparse.ArgumentParser(description="LBAE")
ap.add_argument("--tensorboard_log")
ns = ap.parse_args()

celeba_dataset = torchvision.datasets.ImageFolder(
    root="/home/julian/ImageDataSets/celeba/",
    transform = torchvision.transforms.Compose( [
        torchvision.transforms.Pad( ( -15, -40,-15-1, -30-1) ), torchvision.transforms.Resize( 32 ), torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: dl.quantize(x,8)) ] ))

celeba_dataset = torchvision.datasets.CelebA(
    root="/home/julian/ImageDataSets",
    transform = torchvision.transforms.Compose( [
        torchvision.transforms.Pad((-15, -40,-15-1, -30-1)),
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: dl.quantize(x,8)) ] ) )


class GenderTrainer(Train.LightningTrainer):
    def __init__(self, model, dataset, add_graph=False, learning_rate=.001, batch_size=32):
        super(GenderTrainer, self).__init__( model, dataset, add_graph, learning_rate, batch_size)
        self.attr_index = celeba_dataset.attr_names.index('Male')

    def get_distribution(self, y):
        conditional = y[:,self.attr_index:self.attr_index+1].type(torch.float)
        return self.model(conditional)


class GenderPixelCNNCallback(pl.Callback):
    def __init__(self):
        super(GenderPixelCNNCallback, self).__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        target = torch.arange(start=0,end=20,device=pl_module.device).type(torch.long)
        conditional = (target < 10).float().unsqueeze(1)
        imglist = [pl_module.model(conditional[c:c+1]).sample() for c in range(20)]
        imglist = torch.clip(torch.cat(imglist, axis=0),0.0,1.0)
        trainer.logger.experiment.add_image("epoch_image", torchvision.utils.make_grid(imglist, padding=10, nrow=5 ), trainer.current_epoch, dataformats="CHW")


mymodel = pcnn.PixelCNNLayer(
    event_shape=[3,32,32],
    output_distribution_layer=sp.SpatialIndependentDistributionLayer([3, 32, 32], dl.IndependentQuantizedLayer(num_buckets=8), num_params=30),
    num_conditional=1)


pl.Trainer(fast_dev_run = False, gpus=1, accumulate_grad_batches = 16, default_root_dir=ns.tensorboard_log, callbacks=[
    GenderPixelCNNCallback()]).fit(GenderTrainer( mymodel, celeba_dataset, learning_rate = .001, batch_size = 4 ) )
