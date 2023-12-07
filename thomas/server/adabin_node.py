import numpy as np
import argparse
import torch
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from infer import InferenceHelper  # Modified constructor to take path
from PIL import Image as PILImage


class AdaBinNode(Node):
    def __init__(self, device, in_channel):
        super().__init__("image_subscriber")
        self.subscription = self.create_subscription(
            Image,
            in_channel,
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Image, "depth", 10)
        self.bridge = CvBridge()
        self.device = device

    def listener_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PILImage.fromarray(image)
        image = image.resize((512,512))
        bin_centers, predicted_depth = infer_helper.predict_pil(image)
        msg = self.bridge.cv2_to_imgmsg(predicted_depth[0,0], "32FC1")
        self.publisher.publish(msg)


ap = argparse.ArgumentParser(description="AdaBins Depth Estimator")
ap.add_argument("--device", default="cpu")
ap.add_argument("--in_channel", default="image")  # Correct nomenclature?
ns = ap.parse_args()

infer_helper = InferenceHelper(dataset='nyu', device="cpu", pretrained_path="/home/julian/PyGenBrixProj/AdaBins/pretrained/AdaBins_nyu.pt")
rclpy.init()
image_subscriber = AdaBinNode(ns.device, ns.in_channel)
rclpy.spin(image_subscriber)
image_subscriber.destroy_node()
rclpy.shutdown()

