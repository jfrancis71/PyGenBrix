# Not a standalone program, best executed in a jupyter notebook


import numpy as np
import torch
import torch.nn as nn
import torchvision
import PyGenBrix.utils.dataset_utils as ds_utils
import PyGenBrix.utils.py_train as py_train
import PyGenBrix.hippocampus.localization as loc_mod
import matplotlib.pyplot as plt

train_dataset = ds_utils.SequentialFolderImage("/Users/julian/Google Drive/data_sequential", transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor()]))

streamer = loc_mod.StreamDataset(train_dataset)

localizer = loc_mod.Localization()

features = loc_mod.FeatureTransformDataset(streamer, localizer.scene_recognizer)

def epoch_end_callback(distribution, epoch_num, loss):
    if epoch_num % 10 == 0:
        print("Epoch ", epoch_num, " loss=", loss)
        distribution.test(features[0].unsqueeze(0))

py_train.train_distribution( localizer.hmm, features, 25000, epoch_end_fn=epoch_end_callback, batch_size=1)

#localizer.save("railtrain.pt")

#y = [
    [ ((localizer.hmm.observation_matrix[b1]-features[0][b][good])**2).sum().detach().numpy() for b in range(500)] for b1 in range(35)]

#fig, ax = plt.subplots(figsize=(10, 20))
#plt.imshow(np.exp(-np.array(y)/500), cmap='gray')
