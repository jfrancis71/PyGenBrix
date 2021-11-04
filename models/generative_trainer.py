import argparse
import torchvision
import pytorch_lightning as pl

import PyGenBrix.dist_layers.pixelcnn as pixel_cnn
import PyGenBrix.dist_layers.parallelcnn as parallel_cnn
import PyGenBrix.dist_layers.vdvae as vdvae
import PyGenBrix.models.lb_autoencoder as lbae
import PyGenBrix.Train as Train
import PyGenBrix.models.parser as parser

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

event_shape, dataset = parser.get_dataset(ns)
rv_distribution = parser.get_rv_distribution(ns, event_shape)

if ns.model == "PixelCNN":
    model = pixel_cnn.PixelCNNDistribution(event_shape, rv_distribution, nr_resnet=ns.nr_resnet )
elif ns.model == "ParallelCNN":
    model = parallel_cnn.ParallelCNNDistribution(event_shape, rv_distribution, max_unet_layers=3, num_upsampling_stages=int(ns.parallelcnn_num_upsampling_stages))
    if ns.rv_distribution == "spiq3":
        print("parallelcnn and spiq3 incompatible")
        quit()
elif ns.model == "VDVAE":
    model = vdvae.VDVAE(event_shape, rv_distribution)
elif ns.model == "LBAE":
    model = lbae.LBDistribution()
else:
    print("model not recognized")
    quit()

trainer = Train.LightningDistributionTrainer(model, dataset, learning_rate=ns.lr, batch_size=ns.batch_size)
if ns.model != "LBAE":
    callbacks = [Train.LogSamplesEpochCallback(temperature=1.0), Train.LogSamplesEpochCallback(temperature=0.7)]
    if ns.train_log_freq != 0:
        callbacks.append(Train.LogSamplesTrainingCallback(every_global_step=ns.train_log_freq, temperature=1.0))
        callbacks.append(Train.LogSamplesTrainingCallback(every_global_step=ns.train_log_freq, temperature=0.7))
else:
    callbacks = [Train.LogReconstructionEpochCallback()]
pl.Trainer(fast_dev_run=ns.fast_dev_run, gpus=1, accumulate_grad_batches=ns.accumulate_grad_batches,
    max_epochs=ns.max_epochs, callbacks=callbacks, default_root_dir=ns.tensorboard_log).fit(trainer)
