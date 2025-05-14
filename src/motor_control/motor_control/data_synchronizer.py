#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from message_filters import Subscriber, ApproximateTimeSynchronizer
import json
from pathlib import Path

class DataSynchronizer(Node):
    """
    Node to synchronize encoder and LiDAR data and save them to a file.
    """

    def __init__(self):
        super().__init__('data_synchronizer')

        # Subscribers for encoder and LiDAR data
        self.encoder_sub = Subscriber(self, Odometry, 'robot_odom')
        self.lidar_sub = Subscriber(self, LaserScan, 'lidar_scan_topic')

        # Synchronizer for the messages
        self.sync = ApproximateTimeSynchronizer(
            [self.encoder_sub, self.lidar_sub],
            queue_size=10,
            slop=0.1  # Allowable time difference in seconds
        )
        self.sync.registerCallback(self.sync_callback)

        # File to save synchronized data
        self.data_file = Path('synchronized_data.json')
        self.get_logger().info("DataSynchronizer node started.")

    def sync_callback(self, encoder_msg: Odometry, lidar_msg: LaserScan):
        """
        Callback for synchronized messages.

        Args:
            encoder_msg (Odometry): Encoder data.
            lidar_msg (LaserScan): LiDAR data.
        """
        # Extract relevant data
        encoder_data = {
            "position": {
                "x": encoder_msg.pose.pose.position.x,
                "y": encoder_msg.pose.pose.position.y
            },
            "orientation": {
                "z": encoder_msg.pose.pose.orientation.z,
                "w": encoder_msg.pose.pose.orientation.w
            }
        }

        lidar_data = {
            "ranges": list(lidar_msg.ranges),
            "intensities": list(lidar_msg.intensities)
        }

        timestamp = self.get_clock().now().to_msg().sec

        # Combine data
        synchronized_data = {
            "timestamp": timestamp,
            "encoder_data": encoder_data,
            "lidar_data": lidar_data
        }

        # Save to file
        with open(self.data_file, 'a') as file:
            json.dump(synchronized_data, file)
            file.write('\n')

        self.get_logger().info(f"Synchronized data saved at timestamp {timestamp}.")

def main(args=None):
    rclpy.init(args=args)
    node = DataSynchronizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()