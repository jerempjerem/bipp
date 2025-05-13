#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
import time
import numpy as np
from math import sqrt, sin, cos, atan2, radians, degrees
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Quaternion
import argparse
from pathlib import Path
import json
from simple_pid import PID

ANGLE_TOL = 0.4
# ANGLE_TOL = 1
POS_TOL = 0.01
# MAX_ANG_SPEED = 10
MAX_ANG_SPEED = 40
MAX_LIN_SPEED = 0.05
# MAX_LIN_SPEED = 0.2

KP = 0.5
KI = 0.4
KD = 0

def pi_clip(angle):
    out = np.mod(angle, 2*np.pi)
    return (out - 2*np.pi) if (out > np.pi) else out

class Controller(Node):
    msg_twist = Twist()
    last_run = 0
    interval = 10
    # interval = 30
    # interval = 1
    # interval = 0
    # period = 1
    period = 0.1
    time_paused = 0
    time_res = 0
    paused = False
    index = 0
    number = 0

    checkpoints = [
        [1, 0, 1.5708, 5],     # Avance d'un mètre vers la droite, tourne de 90° (pi/2 radians)
        [1, 1, 3.1416, 5],     # Avance d'un mètre vers le haut, tourne de 90° (pi radians)
        [0, 1, -1.5708, 5],    # Avance d'un mètre vers la gauche, tourne de -90° (-pi/2 radians)
        [0, 0, 0, 5]           # Retour au point de départ, orientation initiale
    ]

    pid_a = PID(
        Kp= 0.625,
        # Ki=1 / period / 64,
        # Ki=0.01,
        Ki=0,
        # Kd=0.05,
        Kd=0,
        setpoint=0,
        sample_time=period / 2,
        output_limits=(np.radians(-MAX_ANG_SPEED), np.radians(MAX_ANG_SPEED)),
        auto_mode=False,
        proportional_on_measurement=False,
        error_map=pi_clip,
        starting_output=0
        )

    current_pose = np.array([0, 0, 0], dtype=float)
    goal = np.array([0, 0, 0], dtype=float)
    def __init__(self, node_name:str='controller', chkpts_file:Path=None):
        super().__init__(node_name)
        self.publisher = self.create_publisher(Twist, 'wheel_instructions_topic', 10)
        self.subscriber = self.create_subscription(Odometry, 
                                              "robot_odom", 
                                              self.odometry_listener_callback, 
                                              10)

        self.msg_twist.linear.x = 0.0
        self.msg_twist.linear.y = 0.0
        self.msg_twist.linear.z = 0.0
        self.msg_twist.angular.x = 0.0
        self.msg_twist.angular.y = 0.0
        self.msg_twist.angular.z = 0.0

        if (chkpts_file is not None) and chkpts_file.exists():
            with open(chkpts_file, 'r') as file:
                self.checkpoints = json.load(file)['pts']

        self.pid_a.reset()

        self.last_run = time.time()

        self.get_logger().info("succesfully started!")

    def odometry_listener_callback(self, msg_odom):
        angle = euler_from_quaternion([0, 0, msg_odom.pose.pose.orientation.z, msg_odom.pose.pose.orientation.w])[2]
        self.current_pose = np.array([msg_odom.pose.pose.position.x,
                             msg_odom.pose.pose.position.y,
                             np.mod(angle, 2*np.pi)])

    def should_exec(self, now):
        if self.paused:
            return False
        elif self.time_res != 0:
            self.time_res = 0
            return (self.time_paused - self.last_run + now - self.time_res) > self.interval
        return (now - self.last_run) > self.interval

    # def out_of_tol(self):
    #     out_of_tol_xy = not (np.allclose(self.current_pose[:2], self.goal[:2], rtol=1e-2, atol=1e-2))
    #     goal_a = (self.goal[2]) if (self.goal[2] < np.pi) else (self.goal[2] - 2 * np.pi)
    #     out_of_tol_a = abs(self.current_pose[2] - goal_a) > np.radians(0.5)
    #     return out_of_tol_xy or out_of_tol_a
    
    def move(self, angle=0, dist=0):
        a_speed = np.clip(angle / self.period, np.radians(-MAX_ANG_SPEED), np.radians(MAX_ANG_SPEED))
        l_speed = np.clip(dist / self.period, -MAX_LIN_SPEED, MAX_LIN_SPEED)
        self.msg_twist.linear.x = float(l_speed)
        self.msg_twist.angular.z = float(a_speed)
        # self.msg_twist.angular.z = float(a_speed * (1 - (abs(l_speed) / MAX_LIN_SPEED * 0.5)))
        self.publisher.publish(self.msg_twist)

    def distance(self):
        diff = self.goal[:2] - self.current_pose[:2]
        return sqrt(diff[0]**2 + diff[1]**2)
    
    def goal_dir(self):
        return atan2(self.goal[1] - self.current_pose[1], self.goal[0] - self.current_pose[0])

    def angle_to_goal(self):
        return self.angle_to_dir(self.goal_dir())
    
    def angle_to_dir(self,abs_a):
        rel_a = np.mod(abs_a - self.current_pose[2], 2*np.pi)
        # rel_a = np.mod(self.current_pose[2] - abs_a, 2*np.pi)
        return (rel_a - 2*np.pi) if (rel_a > np.pi) else (rel_a)
    
    def next_point(self, current_ts):
        self.last_run = current_ts
        self.goal = np.array(self.checkpoints.pop(0))
        self.interval = self.goal[3]
        print(self.goal)

    def step_prep(self, tunings):
        self.number = 0
        self.pid_a.tunings = tunings
        self.pid_a.set_auto_mode(True, 0)

    def move_a_speed(self, dir):
        diff_a = self.pid_a(self.current_pose[2])
        self.move(diff_a, 0)
        rclpy.spin_once(self, timeout_sec=self.period)
        diff_a = self.angle_to_dir(dir)
        return diff_a
    
    def orient_chkpt(self):
        self.step_prep((KP, 0, 0))
        self.pid_a.setpoint = self.goal[2]
        while self.number < 20:
            diff_a = self.move_a_speed(self.goal[2])
            if abs(diff_a) > radians(ANGLE_TOL):
                self.number = 0
            else:
                self.pid_a.tunings = (KP, KI, KD)
                self.number += 1
            print(degrees(diff_a), degrees(self.current_pose[2]), '\t', self.pid_a.components[:2])
        self.pid_a.auto_mode = False
        self.move(0, 0)
    
    def orient_target(self):
        self.step_prep((KP, 0, 0))
        while self.number < 20:
        # while abs(diff_a) > radians(ANGLE_TOL):
            diff_a = self.move_a_speed(self.goal_dir())
            if abs(diff_a) > radians(ANGLE_TOL):
                self.number = 0
            else:
                self.pid_a.tunings = (KP, KI, KD)
                self.number += 1
            print(degrees(diff_a), degrees(self.current_pose[2]), '\t', self.pid_a.components[:])
        self.pid_a.auto_mode = False
        self.move(0, 0)

    def forward_target(self):
        self.step_prep((KP, KI, KD))
        diff_xy = self.distance()
        while self.number < 20:
            self.pid_a.setpoint = (atan2(self.goal[1] - self.current_pose[1], self.goal[0] - self.current_pose[0]))
            diff_a = self.pid_a(self.current_pose[2])
            self.move(diff_a, diff_xy)
            rclpy.spin_once(self, timeout_sec=self.period)
            diff_xy = self.distance()
            diff_a = self.angle_to_goal()
            if abs(diff_xy) > POS_TOL:
                self.number = 0
            else:
                self.number += 1
            print(degrees(diff_a), degrees(self.current_pose[2]), '\t', diff_xy, '\t', self.pid_a.setpoint)
        self.pid_a.auto_mode = False
        self.move(0, 0)

    def loop(self):
        self.goal = np.array(self.checkpoints.pop(0))
        self.interval = self.goal[3]
        print(self.goal)
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)
            now = time.time()
            if (self.should_exec(now)):
                self.orient_target()
                self.forward_target()
                self.orient_chkpt()
                if len(self.checkpoints) == 0:
                    break
                    pass
                else:
                    self.next_point(now)
            
def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", nargs='?', type=Path, default=None)
    p = parser.parse_args()

    controller = Controller(chkpts_file=p.file_path)
    controller.loop()

if __name__ == '__main__':
    main()