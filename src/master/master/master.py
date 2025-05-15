#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from message_filters import ApproximateTimeSynchronizer, Subscriber

import math
import csv


class MasterNode(Node):
    def __init__(self):
        super().__init__('master_node')

        # Subscribers using message_filters
        self.lidar_sub = Subscriber(self, LaserScan, 'scan')
        self.odom_sub = Subscriber(self, Odometry, 'odom')

        # Synchronizer
        self.ts = ApproximateTimeSynchronizer(
            [self.lidar_sub, self.odom_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)

        # CSV file setup
        self.csv_filename = 'synchronized_lidar_odom.csv'
        self.csv_file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # CSV header
        self.csv_writer.writerow([
            'timestamp',
            'lidar_angle_deg',
            'lidar_x',
            'lidar_y',
            'odom_x',
            'odom_y',
            'odom_theta'
        ])

        self.get_logger().info(f"Logging to {self.csv_filename}")

    def synced_callback(self, lidar_msg, odom_msg):
        timestamp = self.get_clock().now().seconds_nanoseconds()[0] + \
                    self.get_clock().now().seconds_nanoseconds()[1] * 1e-9

        # Odom position and heading (theta from quaternion)
        pos = odom_msg.pose.pose.position
        ori = odom_msg.pose.pose.orientation
        theta = self.quaternion_to_yaw(ori)

        # LiDAR ranges to (x, y)
        angle = lidar_msg.angle_min
        for r in lidar_msg.ranges:
            if math.isfinite(r):
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                angle_deg = math.degrees(angle)
                self.csv_writer.writerow([
                    timestamp,
                    round(angle_deg, 2),
                    round(x, 3),
                    round(y, 3),
                    round(pos.x, 3),
                    round(pos.y, 3),
                    round(theta, 3)
                ])
            angle += lidar_msg.angle_increment

        self.get_logger().info(f"Synchronized data at t={timestamp:.2f} logged.")

    def quaternion_to_yaw(self, q):
        """Convert quaternion to yaw (rotation around Z axis)"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MasterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
