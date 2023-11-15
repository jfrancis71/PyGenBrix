import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import brickpi3
import time

class MotorSubscriber(Node):
    def __init__(self):
        super().__init__('motor_subscriber')
        self.subscription = self.create_subscription(
            String,
            'motor',
            self.listener_callback,
            10)
        self.bp = brickpi3.BrickPi3()
        self.speed = 50

    def listener_callback(self, msg):
        command = msg.data
        print("Received: ", msg)
        if command == 'forward':
            self.forward()
        elif command == 'backward':
            self.backward()
        elif command == 'left':
            self.left()
        elif command == 'right':
            self.right()
        elif command == 'off':
            self.off()
        else:
            print("Invalid Command: ", command)

    def forward(self):
        self.bp.set_motor_power(self.bp.PORT_A, self.speed)
        self.bp.set_motor_power(self.bp.PORT_D, self.speed)
        time.sleep(1)
        self.off()

    def backward(self):
        self.bp.set_motor_power(self.bp.PORT_A, -self.speed)
        self.bp.set_motor_power(self.bp.PORT_D, -self.speed)
        time.sleep(1)
        self.off()

    def left(self):
        self.bp.set_motor_power(self.bp.PORT_A, -self.speed)
        self.bp.set_motor_power(self.bp.PORT_D, self.speed)
        time.sleep(1)
        self.off()

    def right(self):
        self.bp.set_motor_power(self.bp.PORT_A, +self.speed)
        self.bp.set_motor_power(self.bp.PORT_D, -self.speed)
        time.sleep(1)
        self.off()

    def off(self):
        self.bp.set_motor_power(self.bp.PORT_A, 0)
        self.bp.set_motor_power(self.bp.PORT_D, 0)

rclpy.init()
motor_subscriber = MotorSubscriber()
rclpy.spin(motor_subscriber)
motor_subscriber.off()
motor_subscriber.destroy_node()
rclpy.shutdown()
