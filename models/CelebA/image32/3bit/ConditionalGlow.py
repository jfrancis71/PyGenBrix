import torch
import torchvision
import pytorch_lightning as pl
import PyGenBrix.dist_layers.glow as GlowDistribution
import PyGenBrix.Train as Train

celeba_dataset = torchvision.datasets.CelebA(
    root="/home/julian/ImageDataSets/",
    transform = torchvision.transforms.Compose( [ torchvision.transforms.Pad( ( -15, -40,-15-1, -30-1) ), torchvision.transforms.Resize( 64 ), torchvision.transforms.ToTensor() ] ) )

class LightningLayerTrainer(Train.LightningTrainer):
    def __init__(self, model, dataset, add_graph=False, learning_rate=.001, batch_size=32):
        super(LightningLayerTrainer, self).__init__( model, dataset, add_graph, learning_rate, batch_size)
        self.attrs = [ celeba_dataset.attr_names.index(name) for name in ['Male','Attractive','Black_Hair','Blond_Hair','Bushy_Eyebrows','Smiling'] ]
    def get_distribution(self, y):
        conditional = y[:,self.attrs].type(torch.float)
        return self.model(conditional)

glow_model = GlowDistribution.GlowLayer(num_conditional=6)

def sample_indx(model, c):
    conditional = torch.tensor([c < 10,0,0,0,0,(c%5)<3], device="cpu").float().unsqueeze(0)
    return model(conditional).sample()

class TensorboardEpochSamplesCallback(pl.Callback):
    def __init__(self):
        super(TensorboardEpochSamplesCallback, self).__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        imglist = [sample_indx(pl_module.model, c) for c in range(20)]
        imglist = torch.clip(torch.cat(imglist, axis=0),0.0,1.0)
        trainer.logger.experiment.add_image("epoch_image", torchvision.utils.make_grid(imglist, padding=10, nrow=5 ), trainer.current_epoch, dataformats="CHW")
        torchvision.utils.save_image(torchvision.utils.make_grid(imglist, padding=10, nrow=5),"/home/julian/Google Drive/conditional_pixelcnn.png")

class TensorboardTrainingSamplesCallback(pl.Callback):
    def __init__(self, every_global_step):
        super(TensorboardTrainingSamplesCallback, self).__init__()
        self.every_global_step = every_global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if (pl_module.global_step % self.every_global_step == 0) and (batch_idx % trainer.accumulate_grad_batches == 0):
            pl_module.eval()
            imglist = [sample_indx(pl_module.model, c) for c in range(20)]
            imglist = torch.clip(torch.cat(imglist, axis=0),0.0,1.0)
            pl_module.train()
            trainer.logger.experiment.add_image("train_image", torchvision.utils.make_grid(imglist, padding=10, nrow=5 ), pl_module.global_step, dataformats="CHW")
            torchvision.utils.save_image(torchvision.utils.make_grid(imglist, padding=10, nrow=5),"/home/julian/Google Drive/conditional_glow.png")

pl.Trainer(fast_dev_run = False, gpus=0, accumulate_grad_batches=2,callbacks=[TensorboardEpochSamplesCallback(),TensorboardTrainingSamplesCallback(1000)]).fit(LightningLayerTrainer(glow_model, celeba_dataset, learning_rate = .0001, batch_size = 8))
