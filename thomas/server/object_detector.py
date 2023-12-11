from torchvision.models import detection
import numpy as np
import argparse
import torch
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import BoundingBox2D
from vision_msgs.msg import ObjectHypothesis
from vision_msgs.msg import ObjectHypothesisWithPose
from vision_msgs.msg import Detection2D
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge


class ImageSubscriber(Node):
    def __init__(self, device, in_channel):
        super().__init__("image_subscriber")
        self.subscription = self.create_subscription(
            Image,
            in_channel,
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Detection2DArray, "object_detection_messages", 10)
        self.render_image_publisher = self.create_publisher(Image, "render_image", 10)
        self.bridge = CvBridge()
        self.device = device

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = cv_image.copy().transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        image = torch.FloatTensor(image)
        image = image.to(self.device)
        detections = model(image)[0]
#        labels = [class_labels[id] for id, score in zip(detections["labels"], detections["scores"]) if score > .9]
#        msg = String()
#        msg.data = ','.join(labels)
#        self.publisher.publish(msg)
#        self.get_logger().info('Publishing: "%s"' % msg.data)
        print("det=", detections)
        det_array = Detection2DArray()
        for label_id, score, box in zip(detections["labels"], detections["scores"], detections["boxes"]):
            if score < .9:
                continue
            box = box.detach().type(torch.int64)
            print("Box=", box)
            b0 = box[0].item()
            print("b-0=", b0)
            cv2.rectangle(cv_image, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), (255, 255, 0), 4)
            det = Detection2D()
            det.header = msg.header
            object_hypothesis_with_pose = ObjectHypothesisWithPose()
            object_hypothesis = ObjectHypothesis()
            object_hypothesis.class_id = class_labels[label_id]
            object_hypothesis.score = score.detach().item()
            object_hypothesis_with_pose.hypothesis = object_hypothesis
            det.results.append(object_hypothesis_with_pose)
            bounding_box = BoundingBox2D()
            bounding_box.center.position.x = float((box[0] + box[2])/2)
            bounding_box.center.position.y = float((box[1] + box[3])/2)
            bounding_box.center.theta = 0.0
            bounding_box.size_x = float(2*(bounding_box.center.position.x - box[0]))
            bounding_box.size_y = float(2*(bounding_box.center.position.y - box[1]))
            det.bbox = bounding_box
            det_array.detections.append(det)
        self.publisher.publish(det_array)
        self.render_image_publisher.publish(self.bridge.cv2_to_imgmsg(cv_image))


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
