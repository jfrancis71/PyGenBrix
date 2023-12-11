import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray



class PersonFollower(Node):
    def __init__(self):
        super().__init__("person_follower")
        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self.subscription = self.create_subscription(
            Detection2DArray,
            'object_detection_messages',
            self.listener_callback, 0)
        twist = Twist()
        twist.linear.x = .5
#        self.publisher.publish(twist)

    def listener_callback(self, msg):
        det_array = msg.detections
        x = None
        for detection in det_array:
            if detection.results[0].hypothesis.class_id == "person":
                x = detection.bbox.center.position.x
                size = detection.bbox.size_x
        if x is not None:
            twist = Twist()
            twist.linear.x = -(size-100)/75
            twist.angular.z = -(x-160)/100
            self.publisher.publish(twist)

rclpy.init()
person_follower = PersonFollower()
rclpy.spin(person_follower)
person_follower.destroy_node()
rclpy.shutdown()

