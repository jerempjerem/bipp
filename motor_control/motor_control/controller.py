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
                self.get_logger().info(f"Points de contrôle chargés: {len(self.checkpoints)}")
        
        # Configuration du contrôleur PID
        self._setup_pid_controller()
        
        # Timer pour les logs périodiques
        self.last_log_time = time.time()
        self.log_interval = 0.1  # 100ms
        
        self.get_logger().info("Contrôleur initialisé avec succès!")

    def _init_control_variables(self):
        """Initialise les variables de contrôle"""
        self.msg_twist = Twist()
        self.last_run = time.time()  # Initialiser avec le temps actuel
        self.interval = 1  # Réduire l'intervalle à 1 seconde
        self.period = 0.1
        self.time_paused = 0
        self.time_res = 0
        self.paused = False
        self.index = 0
        self.number = 0
        self.current_goal_index = 0  # Ajout d'un index pour suivre le point actuel
        
        # Points de contrôle par défaut
        self.checkpoints = [
            [1, 0, 1.5708, 5],     # Avance d'un mètre vers la droite, tourne de 90°
            [1, 1, 3.1416, 5],     # Avance d'un mètre vers le haut, tourne de 180°
            [0, 1, -1.5708, 5],    # Avance d'un mètre vers la gauche, tourne de -90°
            [0, 0, 0, 5]           # Retour au point de départ
        ]
        
        self.current_pose = np.array([0, 0, 0], dtype=float)
        self.goal = np.array([0, 0, 0], dtype=float)
        self.get_logger().info(f"Points de contrôle par défaut: {len(self.checkpoints)}")

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

    def _log_position(self):
        """Log la position actuelle toutes les 100ms"""
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self.get_logger().info(
                f"Position: x={self.current_pose[0]:.3f}m, "
                f"y={self.current_pose[1]:.3f}m, "
                f"θ={degrees(self.current_pose[2]):.1f}°, "
                f"Intervalle={self.interval:.1f}s, "
                f"Point actuel={self.current_goal_index}/{len(self.checkpoints)}"
            )
            self.last_log_time = current_time

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
        
        # Log de la position toutes les 100ms
        self._log_position()

    def should_exec(self, now):
        """
        Détermine si une nouvelle commande doit être exécutée.
        
        Args:
            now (float): Timestamp actuel
            
        Returns:
            bool: True si une nouvelle commande doit être exécutée
        """
        if self.paused:
            self.get_logger().debug("Exécution en pause")
            return False
            
        time_diff = (now - self.last_run)
        self.get_logger().debug(f"Temps écoulé: {time_diff:.2f}s, Intervalle: {self.interval}s")
        
        if time_diff >= self.interval:
            self.get_logger().info(f"Temps écoulé ({time_diff:.2f}s) >= intervalle ({self.interval}s)")
            return True
        return False

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
        
        # Log des commandes
        self.get_logger().debug(
            f"Commande: v={l_speed:.3f}m/s, ω={degrees(a_speed):.1f}°/s"
        )

    def distance(self):
        """
        Calcule la distance jusqu'à la cible.
        
        Returns:
            float: Distance en mètres
        """
        diff = self.goal[:2] - self.current_pose[:2]
        dist = sqrt(diff[0]**2 + diff[1]**2)
        self.get_logger().debug(f"Distance jusqu'à la cible: {dist:.3f}m")
        return dist
    
    def goal_dir(self):
        """
        Calcule la direction vers la cible.
        
        Returns:
            float: Angle en radians
        """
        angle = atan2(
            self.goal[1] - self.current_pose[1],
            self.goal[0] - self.current_pose[0]
        )
        self.get_logger().debug(f"Direction vers la cible: {degrees(angle):.1f}°")
        return angle

    def angle_to_goal(self):
        """
        Calcule l'angle relatif vers la cible.
        
        Returns:
            float: Angle relatif en radians
        """
        angle = self.angle_to_dir(self.goal_dir())
        self.get_logger().debug(f"Angle relatif vers la cible: {degrees(angle):.1f}°")
        return angle
    
    def angle_to_dir(self, abs_a):
        """
        Calcule l'angle relatif vers une direction absolue.
        
        Args:
            abs_a (float): Angle absolu en radians
            
        Returns:
            float: Angle relatif en radians
        """
        rel_a = np.mod(abs_a - self.current_pose[2], 2*np.pi)
        angle = (rel_a - 2*np.pi) if (rel_a > np.pi) else rel_a
        self.get_logger().debug(f"Angle relatif: {degrees(angle):.1f}°")
        return angle
    
    def next_point(self, current_ts):
        """
        Passe au point de contrôle suivant.
        
        Args:
            current_ts (float): Timestamp actuel
        """
        if not self.checkpoints:
            self.get_logger().warn("Aucun point de contrôle disponible")
            return
            
        self.goal = np.array(self.checkpoints.pop(0))
        self.interval = self.goal[3]
        self.current_goal_index += 1
        
        self.get_logger().info(
            f"Nouveau point de contrôle {self.current_goal_index}: "
            f"x={self.goal[0]:.2f}m, y={self.goal[1]:.2f}m, "
            f"θ={degrees(self.goal[2]):.1f}°, intervalle={self.interval}s"
        )
        
        # Réinitialiser le temps de dernière exécution
        self.last_run = current_ts

    def step_prep(self, tunings):
        """
        Prépare une nouvelle étape de contrôle.
        
        Args:
            tunings (tuple): Paramètres PID (Kp, Ki, Kd)
        """
        self.number = 0
        self.pid_a.tunings = tunings
        self.pid_a.set_auto_mode(True, 0)
        self.get_logger().debug(f"Préparation étape avec PID: Kp={tunings[0]}, Ki={tunings[1]}, Kd={tunings[2]}")

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
        self.get_logger().debug(f"Différence d'angle: {degrees(diff_a):.1f}°")
        return diff_a
    
    def orient_chkpt(self):
        """Oriente le robot vers le point de contrôle"""
        self.get_logger().info("Orientation vers le point de contrôle")
        self.step_prep((KP, 0, 0))
        self.pid_a.setpoint = self.goal[2]
        
        while self.number < 20:
            diff_a = self.move_a_speed(self.goal[2])
            if abs(diff_a) > radians(ANGLE_TOL):
                self.number = 0
                self.get_logger().debug(f"Angle hors tolérance: {degrees(diff_a):.1f}° > {degrees(ANGLE_TOL):.1f}°")
            else:
                self.pid_a.tunings = (KP, KI, KD)
                self.number += 1
                self.get_logger().debug(f"Angle dans la tolérance: {degrees(diff_a):.1f}°")
        
        self.pid_a.auto_mode = False
        self.move(0, 0)
        self.get_logger().info("Orientation terminée")
    
    def orient_target(self):
        """Oriente le robot vers la cible"""
        self.get_logger().info("Orientation vers la cible")
        self.step_prep((KP, 0, 0))
        
        while self.number < 20:
            diff_a = self.move_a_speed(self.goal_dir())
            if abs(diff_a) > radians(ANGLE_TOL):
                self.number = 0
                self.get_logger().debug(f"Angle hors tolérance: {degrees(diff_a):.1f}° > {degrees(ANGLE_TOL):.1f}°")
            else:
                self.pid_a.tunings = (KP, KI, KD)
                self.number += 1
                self.get_logger().debug(f"Angle dans la tolérance: {degrees(diff_a):.1f}°")
        
        self.pid_a.auto_mode = False
        self.move(0, 0)
        self.get_logger().info("Orientation terminée")

    def forward_target(self):
        """Fait avancer le robot vers la cible"""
        self.get_logger().info("Avance vers la cible")
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
                self.get_logger().debug(f"Distance hors tolérance: {diff_xy:.3f}m > {POS_TOL:.3f}m")
            else:
                self.number += 1
                self.get_logger().debug(f"Distance dans la tolérance: {diff_xy:.3f}m")
        
        self.pid_a.auto_mode = False
        self.move(0, 0)
        self.get_logger().info("Avance terminée")

    def loop(self):
        """Boucle principale du contrôleur"""
        self.get_logger().info("Démarrage de la boucle principale")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)
            now = time.time()
            
            # Log de l'état toutes les 100ms
            self._log_position()
            
            if self.should_exec(now):
                self.get_logger().info(f"Exécution d'une nouvelle commande (point {self.current_goal_index + 1})")
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
