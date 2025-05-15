#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
    

class MasterNode(Node):
    def __init__(self):
        super().__init__('master_node')
        
        # Subscriber to motor data
        self.motor_subscription = self.create_subscription(
            Odometry,
            'odom',
            self.motor_callback,
            10
        )
        
        # Subscriber to LiDAR data
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10
        )

    def motor_callback(self, msg):
        self.get_logger().info(f'Received motor data: {msg}')
        # Log or process motor data here

    def lidar_callback(self, msg):
        self.get_logger().info(f'Received LiDAR data: {msg.ranges[:5]}')  # Log first 5 ranges
        # Log or process LiDAR data here


def main(args=None):
    rclpy.init(args=args)
    node = MasterNode()
    rclpy.spin(node)
    rclpy.shutdown()
     
     
if __name__ == "__main__":
    main()