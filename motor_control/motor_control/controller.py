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

# Constantes de contrôle
ANGLE_TOL = 0.2  # Tolérance d'angle en radians
POS_TOL = 0.01   # Tolérance de position en mètres
MAX_ANG_SPEED = 40  # Vitesse angulaire maximale en degrés/s
MAX_LIN_SPEED = 0.05  # Vitesse linéaire maximale en m/s

# Paramètres PID
KP = 0.5  # Gain proportionnel
KI = 0.4  # Gain intégral
KD = 0    # Gain dérivé

def pi_clip(angle):
    """
    Normalise un angle entre -π et π.
    
    Args:
        angle (float): Angle en radians
        
    Returns:
        float: Angle normalisé
    """
    out = np.mod(angle, 2*np.pi)
    return (out - 2*np.pi) if (out > np.pi) else out

class Controller(Node):
    """
    Contrôleur de navigation pour le robot.
    Gère le suivi de trajectoire et le contrôle des moteurs.
    """
    
    def __init__(self, node_name:str='controller', chkpts_file:Path=None):
        """
        Initialise le contrôleur.
        
        Args:
            node_name (str): Nom du nœud ROS
            chkpts_file (Path): Chemin vers le fichier de points de contrôle
        """
        super().__init__(node_name)
        
        # Configuration des communications ROS
        self.publisher = self.create_publisher(Twist, 'wheel_instructions_topic', 10)
        self.subscriber = self.create_subscription(
            Odometry, 
            "robot_odom", 
            self.odometry_listener_callback, 
            10
        )
        
        # Initialisation des variables de contrôle
        self._init_control_variables()
        
        # Chargement des points de contrôle
        if (chkpts_file is not None) and chkpts_file.exists():
            with open(chkpts_file, 'r') as file:
                self.checkpoints = json.load(file)['pts']
        
        # Configuration du contrôleur PID
        self._setup_pid_controller()
        
        self.get_logger().info("Contrôleur initialisé avec succès!")

    def _init_control_variables(self):
        """Initialise les variables de contrôle"""
        self.msg_twist = Twist()
        self.last_run = 0
        self.interval = 10
        self.period = 0.1
        self.time_paused = 0
        self.time_res = 0
        self.paused = False
        self.index = 0
        self.number = 0
        
        # Points de contrôle par défaut
        self.checkpoints = [
            [1, 0, 1.5708, 5],     # Avance d'un mètre vers la droite, tourne de 90°
            [1, 1, 3.1416, 5],     # Avance d'un mètre vers le haut, tourne de 180°
            [0, 1, -1.5708, 5],    # Avance d'un mètre vers la gauche, tourne de -90°
            [0, 0, 0, 5]           # Retour au point de départ
        ]
        
        self.current_pose = np.array([0, 0, 0], dtype=float)
        self.goal = np.array([0, 0, 0], dtype=float)

    def _setup_pid_controller(self):
        """Configure le contrôleur PID"""
        self.pid_a = PID(
            Kp=0.625,
            Ki=0,
            Kd=0,
            setpoint=0,
            sample_time=self.period / 2,
            output_limits=(np.radians(-MAX_ANG_SPEED), np.radians(MAX_ANG_SPEED)),
            auto_mode=False,
            proportional_on_measurement=False,
            error_map=pi_clip,
            starting_output=0
        )
        self.pid_a.reset()

    def odometry_listener_callback(self, msg_odom):
        """
        Callback pour les messages d'odométrie.
        
        Args:
            msg_odom (Odometry): Message d'odométrie reçu
        """
        angle = euler_from_quaternion([
            0, 0, 
            msg_odom.pose.pose.orientation.z, 
            msg_odom.pose.pose.orientation.w
        ])[2]
        
        self.current_pose = np.array([
            msg_odom.pose.pose.position.x,
            msg_odom.pose.pose.position.y,
            np.mod(angle, 2*np.pi)
        ])

    def should_exec(self, now):
        """
        Détermine si une nouvelle commande doit être exécutée.
        
        Args:
            now (float): Timestamp actuel
            
        Returns:
            bool: True si une nouvelle commande doit être exécutée
        """
        if self.paused:
            return False
        elif self.time_res != 0:
            self.time_res = 0
            return (self.time_paused - self.last_run + now - self.time_res) > self.interval
        return (now - self.last_run) > self.interval

    def move(self, angle=0, dist=0):
        """
        Envoie une commande de mouvement aux moteurs.
        
        Args:
            angle (float): Angle de rotation en radians
            dist (float): Distance à parcourir en mètres
        """
        # Calcul et limitation des vitesses
        a_speed = np.clip(angle / self.period, np.radians(-MAX_ANG_SPEED), np.radians(MAX_ANG_SPEED))
        l_speed = np.clip(dist / self.period, -MAX_LIN_SPEED, MAX_LIN_SPEED)
        
        # Envoi de la commande
        self.msg_twist.linear.x = float(l_speed)
        self.msg_twist.angular.z = float(a_speed)
        self.publisher.publish(self.msg_twist)

    def distance(self):
        """
        Calcule la distance jusqu'à la cible.
        
        Returns:
            float: Distance en mètres
        """
        diff = self.goal[:2] - self.current_pose[:2]
        return sqrt(diff[0]**2 + diff[1]**2)
    
    def goal_dir(self):
        """
        Calcule la direction vers la cible.
        
        Returns:
            float: Angle en radians
        """
        return atan2(
            self.goal[1] - self.current_pose[1],
            self.goal[0] - self.current_pose[0]
        )

    def angle_to_goal(self):
        """
        Calcule l'angle relatif vers la cible.
        
        Returns:
            float: Angle relatif en radians
        """
        return self.angle_to_dir(self.goal_dir())
    
    def angle_to_dir(self, abs_a):
        """
        Calcule l'angle relatif vers une direction absolue.
        
        Args:
            abs_a (float): Angle absolu en radians
            
        Returns:
            float: Angle relatif en radians
        """
        rel_a = np.mod(abs_a - self.current_pose[2], 2*np.pi)
        return (rel_a - 2*np.pi) if (rel_a > np.pi) else rel_a
    
    def next_point(self, current_ts):
        """
        Passe au point de contrôle suivant.
        
        Args:
            current_ts (float): Timestamp actuel
        """
        self.last_run = current_ts
        self.goal = np.array(self.checkpoints.pop(0))
        self.interval = self.goal[3]
        self.get_logger().info(f"Nouveau point de contrôle: {self.goal}")

    def step_prep(self, tunings):
        """
        Prépare une nouvelle étape de contrôle.
        
        Args:
            tunings (tuple): Paramètres PID (Kp, Ki, Kd)
        """
        self.number = 0
        self.pid_a.tunings = tunings
        self.pid_a.set_auto_mode(True, 0)

    def move_a_speed(self, dir):
        """
        Contrôle la vitesse angulaire.
        
        Args:
            dir (float): Direction cible en radians
            
        Returns:
            float: Différence d'angle
        """
        diff_a = self.pid_a(self.current_pose[2])
        self.move(diff_a, 0)
        rclpy.spin_once(self, timeout_sec=self.period)
        diff_a = self.angle_to_dir(dir)
        return diff_a
    
    def orient_chkpt(self):
        """Oriente le robot vers le point de contrôle"""
        self.step_prep((KP, 0, 0))
        self.pid_a.setpoint = self.goal[2]
        
        while self.number < 20:
            diff_a = self.move_a_speed(self.goal[2])
            if abs(diff_a) > radians(ANGLE_TOL):
                self.number = 0
            else:
                self.pid_a.tunings = (KP, KI, KD)
                self.number += 1
            self.get_logger().debug(f"Angle: {degrees(diff_a)}°, Orientation: {degrees(self.current_pose[2])}°")
        
        self.pid_a.auto_mode = False
        self.move(0, 0)
    
    def orient_target(self):
        """Oriente le robot vers la cible"""
        self.step_prep((KP, 0, 0))
        
        while self.number < 20:
            diff_a = self.move_a_speed(self.goal_dir())
            if abs(diff_a) > radians(ANGLE_TOL):
                self.number = 0
            else:
                self.pid_a.tunings = (KP, KI, KD)
                self.number += 1
            self.get_logger().debug(f"Angle: {degrees(diff_a)}°, Orientation: {degrees(self.current_pose[2])}°")
        
        self.pid_a.auto_mode = False
        self.move(0, 0)

    def forward_target(self):
        """Fait avancer le robot vers la cible"""
        self.step_prep((KP, KI, KD))
        diff_xy = self.distance()
        
        while self.number < 20:
            self.pid_a.setpoint = self.goal_dir()
            diff_a = self.pid_a(self.current_pose[2])
            self.move(diff_a, diff_xy)
            rclpy.spin_once(self, timeout_sec=self.period)
            
            diff_xy = self.distance()
            diff_a = self.angle_to_goal()
            
            if abs(diff_xy) > POS_TOL:
                self.number = 0
            else:
                self.number += 1
            self.get_logger().debug(f"Distance: {diff_xy}m, Angle: {degrees(diff_a)}°")
        
        self.pid_a.auto_mode = False
        self.move(0, 0)

    def loop(self):
        """Boucle principale du contrôleur"""
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)
            now = time.time()
            
            if self.should_exec(now):
                if not self.checkpoints:
                    self.get_logger().info("Trajectoire terminée")
                    break
                    
                self.next_point(now)
                self.orient_chkpt()
                self.forward_target()
                self.orient_target()

def main(args=None):
    """Point d'entrée principal"""
    rclpy.init(args=args)
    controller = Controller("controller")
    controller.loop()

if __name__ == '__main__':
    main()
