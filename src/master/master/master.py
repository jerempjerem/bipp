#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from message_filters import ApproximateTimeSynchronizer, Subscriber


class MasterNode(Node):
    def __init__(self):
        super().__init__('master_node')
        
        # Create message_filters subscribers (not rclpy.create_subscription)
        self.lidar_sub = Subscriber(self, LaserScan, 'scan')
        self.odom_sub = Subscriber(self, Odometry, 'odom')

        # Synchronizer (queue size 10, slop 0.1s tolerance)
        self.ts = ApproximateTimeSynchronizer(
            [self.lidar_sub, self.odom_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)

    def synced_callback(self, lidar_msg, odom_msg):
        self.get_logger().info('Synchronized callback triggered')
        self.get_logger().info(f'LiDAR ranges (first 5): {lidar_msg.ranges[:5]}')
        self.get_logger().info(f'Odom pose: {odom_msg.pose.pose}')


def main(args=None):
    rclpy.init(args=args)
    node = MasterNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
