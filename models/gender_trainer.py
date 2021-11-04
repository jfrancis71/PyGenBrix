import argparse
import torch
import torchvision
import pytorch_lightning as pl

import PyGenBrix.dist_layers.pixelcnn as pixel_cnn
import PyGenBrix.dist_layers.glow as glow
import PyGenBrix.dist_layers.parallelcnn as parallel_cnn
import PyGenBrix.dist_layers.vdvae as vdvae
import PyGenBrix.Train as Train
import PyGenBrix.dist_layers.common_layers as dl
import PyGenBrix.dist_layers.spatial_independent as sp
import PyGenBrix.models.parser as parser


def logging_distribution_samples(pl_module, model, name, gender, current_step, batch_size, temperature=1.0, filename=None):
    imglist = [model(gender*torch.ones([batch_size,1]).cuda()).sample([], temperature) for _ in range(16//batch_size)]
    imglist = torch.clip(torch.cat(imglist, axis=0),0.0,1.0)
    grid_image = torchvision.utils.make_grid(imglist, padding=10, nrow=4 )
    pl_module.logger.experiment.add_image(name+"T"+str(temperature), grid_image, current_step, dataformats="CHW")
    if filename is not None:
        torchvision.utils.save_image(grid_image, filename)


class GenderTrainer(Train.LightningTrainer):
    def __init__(self, model, dataset, add_graph=False, learning_rate=.001, batch_size=32):
        super(GenderTrainer, self).__init__( model, dataset, add_graph, learning_rate, batch_size)
        self.attr_index = dataset.attr_names.index('Male')

    def get_distribution(self, y):
        conditional = y[:,self.attr_index:self.attr_index+1].type(torch.float)
        return self.model(conditional)


class LogSamplesEpochCallback(pl.Callback):
    def __init__(self, temperature=1.0, filename=None):
        super(LogSamplesEpochCallback, self).__init__()
        self.temperature = temperature
        self.filename = filename

    def on_validation_epoch_end(self, trainer, pl_module):
        logging_distribution_samples(pl_module, pl_module.model, "Epoch_Male", 1.0, pl_module.current_epoch, pl_module.batch_size, temperature=self.temperature, filename=self.filename)
        logging_distribution_samples(pl_module, pl_module.model, "Epoch_Female", 0.0, pl_module.current_epoch, pl_module.batch_size, temperature=self.temperature, filename=self.filename)


class LogSamplesTrainingCallback(pl.Callback):
    def __init__(self, every_global_step, temperature=1.0, filename=None):
        super(LogSamplesTrainingCallback, self).__init__()
        self.temperature = temperature
        self.filename = filename
        self.every_global_step = every_global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if (pl_module.global_step % self.every_global_step == 0) and (batch_idx % trainer.accumulate_grad_batches == 0):
            pl_module.eval()
            logging_distribution_samples(pl_module, pl_module.model, "Train_Male", 1.0, pl_module.global_step, pl_module.batch_size, temperature=self.temperature, filename=self.filename)
            logging_distribution_samples(pl_module, pl_module.model, "Train_Female", 0.0, pl_module.global_step, pl_module.batch_size, temperature=self.temperature, filename=self.filename)
            pl_module.train()


ap = argparse.ArgumentParser(description="conditional_gender trainer")
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
ap.add_argument("--filename", default=None)
ns = ap.parse_args()

if ns.dataset != "celeba32" and ns.dataset != "celeba64":
    print("This program only supports celeba")
    quit()
event_shape, dataset = parser.get_dataset(ns)
rv_distribution = parser.get_rv_distribution(ns, event_shape)


if ns.model == "PixelCNN":
    model = pixel_cnn.PixelCNNLayer(event_shape, rv_distribution, 1, nr_resnet=ns.nr_resnet )
elif ns.model == "Glow":
    model = glow.GlowLayer(num_conditional=1)
else:
    print("model not recognized")
    quit()

trainer = GenderTrainer(model, dataset, learning_rate = ns.lr, batch_size = ns.batch_size)
callbacks = [LogSamplesEpochCallback(temperature=1.0), LogSamplesEpochCallback(temperature=0.7)]
if ns.train_log_freq != 0:
    callbacks.append(LogSamplesTrainingCallback(every_global_step=ns.train_log_freq, temperature=1.0, filename=ns.filename))
    callbacks.append(LogSamplesTrainingCallback(every_global_step=ns.train_log_freq, temperature=0.7, filename=ns.filename))
pl.Trainer(fast_dev_run=ns.fast_dev_run, gpus=1, accumulate_grad_batches=ns.accumulate_grad_batches, max_epochs=ns.max_epochs, 
           callbacks=callbacks,
           default_root_dir=ns.tensorboard_log).fit(trainer)
