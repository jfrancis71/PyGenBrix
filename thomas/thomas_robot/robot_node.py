import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import brickpi3
import time

class RobotSubscriber(Node):
    def __init__(self):
        super().__init__('robot_subscriber')
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.listener_callback,
            10)
        self.bp = brickpi3.BrickPi3()
        self.speed = 50
        print("Robot ready.")

    def listener_callback(self, msg):
        print("Callback message: ", msg)
        motora = msg.linear.x*25
        motorb = msg.linear.x*25
        motora += msg.angular.z * 10
        motorb -= msg.angular.z * 10
        self.bp.set_motor_power(self.bp.PORT_A, motora)
        self.bp.set_motor_power(self.bp.PORT_D, motorb)
        time.sleep(1)
        self.off()

    def off(self):
        self.bp.set_motor_power(self.bp.PORT_A, 0)
        self.bp.set_motor_power(self.bp.PORT_D, 0)

rclpy.init()
motor_subscriber = RobotSubscriber()
rclpy.spin(motor_subscriber)
motor_subscriber.off()
motor_subscriber.destroy_node()
rclpy.shutdown()
