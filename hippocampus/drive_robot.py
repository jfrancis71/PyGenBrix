import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
import rpyc
import PIL
import torch
import torchvision
import PyGenBrix.hippocampus.localization as localization
import PyGenBrix.hippocampus.scene_recognition as scene_recognition
import PyGenBrix.utils.dataset_utils as ds


class Static_HMM:
    def __init__(self):
        self.state_distribution = torch.ones([35])/35
        self.transition_matrix = 0.1*torch.eye(35) + .8*torch.roll(torch.eye(35), 1,1) + .08*torch.roll(torch.eye(35), 2,1) + (.02/35)*torch.ones([35,35])
        torch.manual_seed(42)
        self.scene = scene_recognition.SceneRecognition()
        train_dataset = ds.SequentialFolderImage(ns.data_folder, transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor()]))
#        print("Train dataset0", train_dataset[0][0][0,:,0])
        self.train_tensor = torch.stack([train_dataset[t][0][0] for t in range(50)])
        self.features = self.scene(self.train_tensor)
#        print("REP1=", self.features[5,:,6,7])
        
    def observe(self, image_tensor):
#        image_tensor = self.train_tensor[12:13]
#        print("Image tensor=", image_tensor[0,5])
        inp = image_tensor[:1]
#        print("Input shape=", inp.shape)
        representation = self.scene(image_tensor)
#        print("image rep=", representation[0,:,5,6])
        observation_values = np.array([torch.sum((representation-self.features[po])**2) for po in range(35)])
        observation_vector = np.exp(-observation_values/500.)
#        print("obs=", observation_vector)
        unnormalized = torch.matmul(self.state_distribution, self.transition_matrix)*observation_vector
        self.state_distribution = unnormalized/torch.sum(unnormalized)


ap = argparse.ArgumentParser(description="drive_robot")
ap.add_argument("--ip_address") # ip address of Pi
ap.add_argument("--data_folder", default="data")
ap.add_argument("--learned_model", default=".")
ap.add_argument("--model", default="static_hmm")
ns = ap.parse_args()
conn = rpyc.classic.connect(ns.ip_address)
conn.execute("from ev3dev2.motor import LargeMotor, OUTPUT_A")
conn.execute("m = LargeMotor(OUTPUT_A)")
seq = 0
if ns.model == "static_hmm":
    model = Static_HMM()
elif ns.model == "learned_hmm":
    model = localization.Localization()
    model.load(ns.learned_model)
print("model feature=", model.features.shape)
y_pos = np.arange(35)
plt.ion()
figure, ax = plt.subplots(figsize=(6, 4))
performance = [1]*35
ln = ax.bar(y_pos, performance, align='center', alpha=0.5)
while True:
    print("START IMPORT")
    image = ds.import_http_image("http://"+ns.ip_address+":8080?action=snapshot")
    print("END IMPORT")
    image_tensor=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor()])(image)
    model.observe(image_tensor)
    for i in range(35):
        ln[i].set_height(model.state_distribution[i])
        figure.canvas.draw()
        figure.canvas.flush_events()
    print(model.state_distribution)
    conn.execute("m.on_for_rotations(75, 2)")
    seq += 1
