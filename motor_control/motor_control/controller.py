#!/usr/bin/env python3
"""
Robot controller node for navigation and movement control.
This module implements a PID-based controller for robot navigation through checkpoints.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped, Quaternion
from nav_msgs.msg import Odometry
import time
import numpy as np
from math import sqrt, sin, cos, atan2, radians, degrees
from tf_transformations import quaternion_from_euler, euler_from_quaternion
import argparse
from pathlib import Path
import json
from simple_pid import PID

# Constants for robot control
ANGLE_TOLERANCE = 0.4  # Angle tolerance in radians
POSITION_TOLERANCE = 0.01  # Position tolerance in meters
MAX_ANGULAR_SPEED = 40  # Maximum angular speed in degrees/s
MAX_LINEAR_SPEED = 0.05  # Maximum linear speed in m/s
ODOMETRY_LOG_INTERVAL = 0.2  # Log odometry every 200ms

# PID controller constants
KP = 0.5  # Proportional gain
KI = 0.4  # Integral gain
KD = 0.0  # Derivative gain

def pi_clip(angle: float) -> float:
    """
    Normalize angle to [-pi, pi] range.
    
    Args:
        angle (float): Input angle in radians
        
    Returns:
        float: Normalized angle in [-pi, pi] range
    """
    out = np.mod(angle, 2 * np.pi)
    return (out - 2 * np.pi) if (out > np.pi) else out

class Controller(Node):
    """
    Robot controller class implementing navigation and movement control.
    Uses PID control for both orientation and position control.
    """
    
    def __init__(self, node_name: str = 'controller', chkpts_file: Path = None):
        """
        Initialize the controller node.
        
        Args:
            node_name (str): Name of the ROS2 node
            chkpts_file (Path): Path to checkpoint configuration file
        """
        super().__init__(node_name)
        
        # Initialize ROS publishers and subscribers
        self.publisher = self.create_publisher(Twist, 'wheel_instructions_topic', 10)
        self.subscriber = self.create_subscription(
            Odometry, 
            "robot_odom", 
            self.odometry_listener_callback, 
            10
        )

        # Initialize control parameters
        self.msg_twist = Twist()
        self.last_run = 0
        self.interval = 10
        self.period = 0.1
        self.time_paused = 0
        self.time_res = 0
        self.paused = False
        self.index = 0
        self.number = 0
        self.last_odometry_log = 0
        self.checkpoint_start_time = time.time()

        # Default checkpoints if no file provided
        self.checkpoints = [
            [1, 0, 1.5708, 5],     # Move 1m right, rotate 90°
            [1, 1, 3.1416, 5],     # Move 1m up, rotate 180°
            [0, 1, -1.5708, 5],    # Move 1m left, rotate -90°
            [0, 0, 0, 5]           # Return to start, initial orientation
        ]

        # Initialize PID controller for angular control
        self.pid_a = PID(
            Kp=0.625,
            Ki=0,
            Kd=0,
            setpoint=0,
            sample_time=self.period / 2,
            output_limits=(np.radians(-MAX_ANGULAR_SPEED), np.radians(MAX_ANGULAR_SPEED)),
            auto_mode=False,
            proportional_on_measurement=False,
            error_map=pi_clip,
            starting_output=0
        )

        # Initialize robot state
        self.current_pose = np.array([0, 0, 0], dtype=float)
        self.goal = np.array([0, 0, 0], dtype=float)

        # Load checkpoints from file if provided
        if (chkpts_file is not None) and chkpts_file.exists():
            with open(chkpts_file, 'r') as file:
                self.checkpoints = json.load(file)['pts']

        self.pid_a.reset()
        self.last_run = time.time()
        self.get_logger().info("Controller successfully started!")

    def odometry_listener_callback(self, msg_odom: Odometry) -> None:
        """
        Callback for odometry messages.
        Updates the current pose of the robot and logs odometry data.
        
        Args:
            msg_odom (Odometry): Odometry message containing robot pose
        """
        angle = euler_from_quaternion([
            0, 0, 
            msg_odom.pose.pose.orientation.z, 
            msg_odom.pose.pose.orientation.w
        ])[2]
        self.current_pose = np.array([
            msg_odom.pose.pose.position.x,
            msg_odom.pose.pose.position.y,
            np.mod(angle, 2 * np.pi)
        ])

        # Log odometry data every ODOMETRY_LOG_INTERVAL seconds
        current_time = time.time()
        if current_time - self.last_odometry_log >= ODOMETRY_LOG_INTERVAL:
            elapsed_time = current_time - self.checkpoint_start_time
            self.get_logger().info(
                f"Odometry - Position: ({self.current_pose[0]:.3f}, {self.current_pose[1]:.3f})m, "
                f"Orientation: {degrees(self.current_pose[2]):.1f}°, "
                f"Time since checkpoint: {elapsed_time:.1f}s"
            )
            self.last_odometry_log = current_time

    def should_exec(self, now: float) -> bool:
        """
        Check if the controller should execute the next movement.
        
        Args:
            now (float): Current timestamp
            
        Returns:
            bool: True if controller should execute, False otherwise
        """
        if self.paused:
            return False
        elif self.time_res != 0:
            self.time_res = 0
            return (self.time_paused - self.last_run + now - self.time_res) > self.interval
        return (now - self.last_run) > self.interval

    def move(self, angle: float = 0, dist: float = 0) -> None:
        """
        Send movement commands to the robot.
        
        Args:
            angle (float): Angular velocity in radians/s
            dist (float): Linear velocity in m/s
        """
        a_speed = np.clip(
            angle / self.period, 
            np.radians(-MAX_ANGULAR_SPEED), 
            np.radians(MAX_ANGULAR_SPEED)
        )
        l_speed = np.clip(dist / self.period, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED)
        
        self.msg_twist.linear.x = float(l_speed)
        self.msg_twist.angular.z = float(a_speed)
        self.publisher.publish(self.msg_twist)

    def distance(self) -> float:
        """
        Calculate Euclidean distance to current goal.
        
        Returns:
            float: Distance to goal in meters
        """
        diff = self.goal[:2] - self.current_pose[:2]
        return sqrt(diff[0]**2 + diff[1]**2)
    
    def goal_dir(self) -> float:
        """
        Calculate absolute angle to goal.
        
        Returns:
            float: Angle to goal in radians
        """
        return atan2(self.goal[1] - self.current_pose[1], 
                    self.goal[0] - self.current_pose[0])

    def angle_to_goal(self) -> float:
        """
        Calculate relative angle to goal.
        
        Returns:
            float: Relative angle to goal in radians
        """
        return self.angle_to_dir(self.goal_dir())
    
    def angle_to_dir(self, abs_a: float) -> float:
        """
        Convert absolute angle to relative angle.
        
        Args:
            abs_a (float): Absolute angle in radians
            
        Returns:
            float: Relative angle in radians
        """
        rel_a = np.mod(abs_a - self.current_pose[2], 2 * np.pi)
        return (rel_a - 2 * np.pi) if (rel_a > np.pi) else rel_a
    
    def next_point(self, current_ts: float) -> None:
        """
        Set next checkpoint as goal.
        
        Args:
            current_ts (float): Current timestamp
        """
        self.last_run = current_ts
        self.goal = np.array(self.checkpoints.pop(0))
        self.interval = self.goal[3]
        self.checkpoint_start_time = time.time()  # Reset checkpoint timer
        self.get_logger().info(f"Moving to next checkpoint: {self.goal}")

    def step_prep(self, tunings: tuple) -> None:
        """
        Prepare PID controller for next movement step.
        
        Args:
            tunings (tuple): PID controller tunings (Kp, Ki, Kd)
        """
        self.number = 0
        self.pid_a.tunings = tunings
        self.pid_a.set_auto_mode(True, 0)

    def move_a_speed(self, dir: float) -> float:
        """
        Move robot with angular speed control.
        
        Args:
            dir (float): Target direction in radians
            
        Returns:
            float: Angular error in radians
        """
        diff_a = self.pid_a(self.current_pose[2])
        self.move(diff_a, 0)
        rclpy.spin_once(self, timeout_sec=self.period)
        diff_a = self.angle_to_dir(dir)
        return diff_a
    
    def orient_chkpt(self) -> None:
        """
        Orient robot to checkpoint orientation.
        """
        self.step_prep((KP, 0, 0))
        self.pid_a.setpoint = self.goal[2]
        while self.number < 20:
            diff_a = self.move_a_speed(self.goal[2])
            if abs(diff_a) > radians(ANGLE_TOLERANCE):
                self.number = 0
            else:
                self.pid_a.tunings = (KP, KI, KD)
                self.number += 1
            self.get_logger().debug(
                f"Orientation: {degrees(diff_a):.1f}°, "
                f"Current: {degrees(self.current_pose[2]):.1f}°, "
                f"PID: {self.pid_a.components[:2]}"
            )
        self.pid_a.auto_mode = False
        self.move(0, 0)
    
    def orient_target(self) -> None:
        """
        Orient robot towards target position.
        """
        self.step_prep((KP, 0, 0))
        while self.number < 20:
            diff_a = self.move_a_speed(self.goal_dir())
            if abs(diff_a) > radians(ANGLE_TOLERANCE):
                self.number = 0
            else:
                self.pid_a.tunings = (KP, KI, KD)
                self.number += 1
            self.get_logger().debug(
                f"Target orientation: {degrees(diff_a):.1f}°, "
                f"Current: {degrees(self.current_pose[2]):.1f}°, "
                f"PID: {self.pid_a.components}"
            )
        self.pid_a.auto_mode = False
        self.move(0, 0)

    def forward_target(self) -> None:
        """
        Move robot forward towards target position.
        """
        self.step_prep((KP, KI, KD))
        diff_xy = self.distance()
        while self.number < 20:
            self.pid_a.setpoint = self.goal_dir()
            diff_a = self.pid_a(self.current_pose[2])
            self.move(diff_a, diff_xy)
            rclpy.spin_once(self, timeout_sec=self.period)
            diff_xy = self.distance()
            diff_a = self.angle_to_goal()
            if abs(diff_xy) > POSITION_TOLERANCE:
                self.number = 0
            else:
                self.number += 1
            self.get_logger().debug(
                f"Position: {diff_xy:.3f}m, "
                f"Angle: {degrees(diff_a):.1f}°, "
                f"Current: {degrees(self.current_pose[2]):.1f}°, "
                f"Target: {degrees(self.pid_a.setpoint):.1f}°"
            )
        self.pid_a.auto_mode = False
        self.move(0, 0)

    def loop(self) -> None:
        """
        Main control loop.
        Executes navigation through checkpoints.
        """
        self.goal = np.array(self.checkpoints.pop(0))
        self.interval = self.goal[3]
        self.checkpoint_start_time = time.time()  # Initialize checkpoint timer
        self.get_logger().info(f"Starting navigation with goal: {self.goal}")
        
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)
            now = time.time()
            if self.should_exec(now):
                self.orient_target()
                self.forward_target()
                self.orient_chkpt()
                if len(self.checkpoints) == 0:
                    self.get_logger().info("Navigation completed!")
                    break
                else:
                    self.next_point(now)

def main(args=None):
    """
    Main entry point for the controller node.
    
    Args:
        args: Command line arguments
    """
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description="Robot controller node")
    parser.add_argument("file_path", nargs='?', type=Path, default=None,
                       help="Path to checkpoint configuration file")
    args = parser.parse_args()

    controller = Controller(chkpts_file=args.file_path)
    controller.loop()

if __name__ == '__main__':
    main()