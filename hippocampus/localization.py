import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
import rpyc
import PIL
import torch
import torchvision
import PyGenBrix.hippocampus.scene_recognition as scene_recognition
import PyGenBrix.utils.dataset_utils as ds


class SingleFolderImage():
    def __init__(self, root, transform):
        self.main_dir = root
        self.transform = transform
        self.num_images = len(os.listdir(root))

    def __len__(self):
        return self.num_images-1

    def __getitem__(self, idx):
        img_loc1 = os.path.join(self.main_dir, "file_" + str(idx)+".jpg")
        image1 = PIL.Image.open(img_loc1).convert("RGB")
        tensor_image1 = self.transform(image1)
        img_loc2 = os.path.join(self.main_dir, "file_" + str(idx+1)+".jpg")
        image2 = PIL.Image.open(img_loc2).convert("RGB")
        tensor_image2 = self.transform(image2)
        return tensor_image1, tensor_image2


ap = argparse.ArgumentParser(description="gather_data")
ap.add_argument("--ip_address") # ip address of Pi
ap.add_argument("--folder", default="data")
ns = ap.parse_args()
train_dataset = SingleFolderImage(ns.folder, transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor()]))
train_tensor = torch.stack([train_dataset[t][0][0] for t in range(50)])
scene = scene_recognition.SceneRecognition()
representations = scene(train_tensor)
conn = rpyc.classic.connect(ns.ip_address)
conn.execute("from ev3dev2.motor import LargeMotor, OUTPUT_A")
conn.execute("m = LargeMotor(OUTPUT_A)")
seq = 0
while True:
    image = ds.import_http_image("http://"+ns.ip_address+":8080?action=snapshot")
    image_tensor=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor()])(image)
    representation = scene(image_tensor[:1])
    res = np.array([torch.sum((representation-representations[po])**2) for po in range(35)])
    print(res.argmin())
    conn.execute("m.on_for_rotations(75, 2)")
    seq += 1
