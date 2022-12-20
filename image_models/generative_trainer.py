import argparse
import torchvision
import pytorch_lightning as pl

import PyGenBrix.utils.trainer as Train
import PyGenBrix.utils.ema_trainer as ema_trainer
import PyGenBrix.image_models.parser as parser

ap = argparse.ArgumentParser(description="generative_trainer")
ap.add_argument("--tensorboard_log")
ap.add_argument("--max_epochs", default=10, type=int)
ap.add_argument("--lr", default=.0002, type=float)
ap.add_argument("--ema_rate", default=.99, type=float)
ap.add_argument("--batch_size", default=8, type=int)
ap.add_argument("--accumulate_grad_batches", default=8, type=int)
ap.add_argument("--fast_dev_run", action="store_true")
ap.add_argument("--dataset")
ap.add_argument("--rv_distribution")
ap.add_argument("--model")
ap.add_argument("--trainer", default="pl")
ap.add_argument("--parallelcnn_num_upsampling_stages", default=1, type=int)
ap.add_argument("--nr_resnet", default=5, type=int)
ap.add_argument("--train_log_freq", default=0, type=int)
ns = ap.parse_args()

event_shape, dataset = parser.get_dataset(ns)
rv_distribution = parser.get_rv_distribution(ns, event_shape)
model = parser.get_model(ns, event_shape, rv_distribution)

if ns.model != "LBAE":
    callbacks = [Train.LogSamplesEpochCallback(temperature=1.0), Train.LogSamplesEpochCallback(temperature=0.7)]
    if ns.train_log_freq != 0:
        callbacks.append(Train.LogSamplesTrainingCallback(every_global_step=ns.train_log_freq, temperature=1.0))
        callbacks.append(Train.LogSamplesTrainingCallback(every_global_step=ns.train_log_freq, temperature=0.7))
else:
    callbacks = [Train.LogReconstructionEpochCallback()]

if ns.trainer == "pl":
    trainer = Train.LightningDistributionTrainer(model, dataset, learning_rate=ns.lr, batch_size=ns.batch_size)
    accumulate_grad_batches = ns.accumulate_grad_batches
elif ns.trainer == "ema":
    train_model = parser.get_model(ns, event_shape, rv_distribution)
    trainer = ema_trainer.EMATrainer(model, train_model, dataset, batch_size=ns.batch_size, learning_rate=ns.lr, ema_rate=ns.ema_rate, accumulate_grad_batches=ns.accumulate_grad_batches)
    accumulate_grad_batches = 1
    model.requires_grad = False

pl.Trainer(fast_dev_run=ns.fast_dev_run, gpus=1, accumulate_grad_batches=accumulate_grad_batches,
    max_epochs=ns.max_epochs, callbacks=callbacks, default_root_dir=ns.tensorboard_log).fit(trainer)
