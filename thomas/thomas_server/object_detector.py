from torchvision.models import detection
import numpy as np
import argparse
import torch
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageSubscriber(Node):
    def __init__(self, device, in_channel):
        super().__init__("image_subscriber")
        self.subscription = self.create_subscription(
            Image,
            in_channel,
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.device = device

    def listener_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        image = torch.FloatTensor(image)
        image = image.to(self.device)
        detections = model(image)[0]
        print("Detections: ", [class_labels[id] for id, score in zip(detections["labels"], detections["scores"]) if score > .9])

ap = argparse.ArgumentParser(description="MobileNet Object Detector")
ap.add_argument("--device", default="cpu")
ap.add_argument("--in_channel", default="image")  # Correct nomenclature?
ns = ap.parse_args()

model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, progress=True,
	pretrained_backbone=True).to(ns.device)
model.eval()
class_labels = detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT.meta["categories"]
rclpy.init()
image_subscriber = ImageSubscriber(ns.device, ns.in_channel)
rclpy.spin(image_subscriber)
image_subscriber.destroy_node()
rclpy.shutdown()
