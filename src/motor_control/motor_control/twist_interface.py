#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import serial
from geometry_msgs.msg import Twist, Quaternion
from nav_msgs.msg import Odometry
from struct import unpack
from numpy import sin, cos, pi, clip
from tf_transformations import quaternion_from_euler

# ===============Command Bits Reference===========================
# const uint8_t m_dir = 1<<0; // Rotation direction 0:backward 1:forward
# const uint8_t m_free = 1<<1; // Free wheel 0:locked 1:freewheel
# const uint8_t m_right = 1<<2; // Apply to right motor 0:no 1:yes
# const uint8_t m_left = 1<<3; // Apply to left motor 0:no 1:yes
# const uint8_t m_rst = 1<<7; // Reset encoder counters
# ================================================================

# Constantes du système
TICKS_REV = 3840  # Nombre de ticks par révolution des encodeurs
TICKS_HALF_REV = TICKS_REV / 2
WHEEL_RADIUS = 0.032  # Rayon des roues en mètres
WHEELS_SEP = 0.225  # Distance entre les roues en mètres
CONTROL_LOOP_PERIOD = 10  # Période de contrôle en millisecondes
MAX_QUEUE_SIZE = 10  # Taille maximale de la file d'attente des messages

class TwistIF(Node):
    """
    Interface entre ROS 2 et les moteurs du robot.
    Gère la communication série, la conversion des commandes et l'odométrie.
    """
    
    def __init__(self, node_name:str='twist_interface'):
        """
        Initialise l'interface avec les paramètres ROS et la communication série.
        
        Args:
            node_name (str): Nom du nœud ROS
        """
        super().__init__(node_name)
        
        # Déclaration des paramètres ROS
        self.declare_parameters(
            namespace='',
            parameters=[
                ('device', '/dev/ttyACM0'),
                ('wheel_instructions_topic', 'wheel_instructions_topic'),
            ]
        )
        
        # Initialisation des variables
        self.wheel_topic_name = self.get_parameter('wheel_instructions_topic').value
        self.device_name = self.get_parameter('device').value
        
        # Configuration de la communication série
        self.ser = serial.Serial(
            self.device_name,
            115200,
            timeout=.1
        )
        
        # Initialisation des variables d'odométrie
        self._init_odometry_variables()
        
        # Configuration des publishers/subscribers ROS
        self._setup_ros_communication()
        
        self.get_logger().info("Interface initialisée avec succès!")

    def _init_odometry_variables(self):
        """Initialise les variables d'odométrie"""
        self.l_speed = self.r_speed = 0
        self.l_odom = self.r_odom = 0
        self.last_l_odom = self.last_r_odom = 0
        self.last_now = 0
        self.x_odom = self.y_odom = self.a_odom = 0
        self.msg_q = []
        self.msg_odom = self._create_odometry_message()

    def _create_odometry_message(self) -> Odometry:
        """Crée et initialise un message d'odométrie"""
        msg = Odometry()
        msg.pose.pose.position.x = 0.0
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        msg.twist.twist.linear.x = 0.0
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.linear.z = 0.0
        msg.twist.twist.angular.x = 0.0
        msg.twist.twist.angular.y = 0.0
        msg.twist.twist.angular.z = 0.0
        return msg

    def _setup_ros_communication(self):
        """Configure les publishers et subscribers ROS"""
        self.subscriber = self.create_subscription(
            Twist, 
            self.wheel_topic_name, 
            self.twist_listener_callback, 
            10
        )
        self.publisher = self.create_publisher(Odometry, 'robot_odom', 10)
        self.ser.reset_input_buffer()

    def serial_read_bytes(self, length=8):
        """
        Lit les données des encodeurs via le port série.
        
        Args:
            length (int): Nombre d'octets à lire
        """
        sermsg = self.ser.readline()
        while len(sermsg) <= length:
            sermsg += self.ser.readline()
        self.l_odom = unpack('<l', sermsg[0:4])[0]
        self.r_odom = unpack('<l', sermsg[4:8])[0]

    def publish_odom(self):
        """Calcule et publie les données d'odométrie"""
        # Calcul des distances parcourues
        l_ticks = self.l_odom - self.last_l_odom
        r_ticks = self.r_odom - self.last_r_odom
        
        # Conversion des ticks en distances
        l_dist = WHEEL_RADIUS * 2 * pi * l_ticks / TICKS_REV
        r_dist = WHEEL_RADIUS * 2 * pi * r_ticks / TICKS_REV
        avg_dist = (l_dist + r_dist) / 2
        
        # Calcul de la rotation
        deltaA = (l_dist - r_dist) / WHEELS_SEP
        self.a_odom += deltaA
        
        # Mise à jour de la position
        deltaX = cos(self.a_odom) * avg_dist
        deltaY = sin(self.a_odom) * avg_dist
        self.x_odom += deltaX
        self.y_odom += deltaY
        
        # Calcul des vitesses
        now = self.get_clock().now()
        interval = (now.nanoseconds / 1000000000) - self.last_now
        
        # Mise à jour du message d'odométrie
        self._update_odometry_message(now, deltaX/interval, deltaY/interval, deltaA/interval)
        
        # Mise à jour des valeurs précédentes
        self.last_l_odom = self.l_odom
        self.last_r_odom = self.r_odom
        self.last_now = now.nanoseconds / 1000000000

    def _update_odometry_message(self, now, x_vel, y_vel, a_vel):
        """
        Met à jour le message d'odométrie avec les nouvelles valeurs.
        
        Args:
            now: Timestamp actuel
            x_vel: Vitesse linéaire en x
            y_vel: Vitesse linéaire en y
            a_vel: Vitesse angulaire
        """
        self.msg_odom.header.stamp = now.to_msg()
        self.msg_odom.pose.pose.position.x = self.x_odom
        self.msg_odom.pose.pose.position.y = self.y_odom
        q = quaternion_from_euler(0, 0, self.a_odom)
        self.msg_odom.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.msg_odom.twist.twist.linear.x = x_vel
        self.msg_odom.twist.twist.linear.y = y_vel
        self.msg_odom.twist.twist.angular.z = a_vel
        self.publisher.publish(self.msg_odom)

    def twist_translator(self, msg):
        """
        Convertit les commandes Twist en commandes moteur.
        
        Args:
            msg (Twist): Message de commande ROS
        """
        # Calcul des vitesses de roue
        avgVel = msg.linear.x
        deltaVel = msg.angular.z * WHEELS_SEP / 2
        dl = avgVel + deltaVel
        dr = avgVel - deltaVel
        
        # Conversion en ticks
        l_ticks = TICKS_REV * dl / (WHEEL_RADIUS * 2 * pi) * CONTROL_LOOP_PERIOD / 1000
        r_ticks = TICKS_REV * dr / (WHEEL_RADIUS * 2 * pi) * CONTROL_LOOP_PERIOD / 1000
        
        # Limitation des vitesses
        self.l_speed = clip(int(l_ticks), -255, 255)
        self.r_speed = clip(int(r_ticks), -255, 255)

    def send_serial_cmd(self):
        """Envoie les commandes aux moteurs via le port série"""
        if self.l_speed == self.r_speed:
            # Commande unique si les deux moteurs ont la même vitesse
            bits, speed = self.cmd_bits_abs_speed(self.l_speed)
            self.ser.write(bytes([0b00001100 | bits, speed]))
        else:
            # Commandes séparées pour chaque moteur
            bits, speed = self.cmd_bits_abs_speed(self.l_speed)
            self.ser.write(bytes([0b00001000 | bits, speed]))
            bits, speed = self.cmd_bits_abs_speed(self.r_speed)
            self.ser.write(bytes([0b00000100 | bits, speed]))

    def cmd_bits_abs_speed(self, speed):
        """
        Génère les bits de commande et la vitesse absolue.
        
        Args:
            speed (int): Vitesse du moteur (-255 à 255)
            
        Returns:
            tuple: (bits de commande, vitesse absolue)
        """
        if speed < 0:
            return 0b00000000, abs(speed)
        elif speed == 0:
            return 0b00000011, 0
        else:
            return 0b00000001, abs(speed)

    def loop(self):
        """Boucle principale de l'interface"""
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)
            self.serial_read_bytes(8)
            self.publish_odom()
            
            # Traitement des commandes en attente
            if self.msg_q:
                self.twist_translator(self.msg_q.pop(0))
                self.send_serial_cmd()
                
                # Nettoyage de la file si trop de messages
                if len(self.msg_q) > MAX_QUEUE_SIZE:
                    self.msg_q.clear()
                    self.get_logger().warn(f'File de messages purgée (taille: {len(self.msg_q)})')

    def twist_listener_callback(self, msg):
        """
        Callback pour les messages Twist entrants.
        
        Args:
            msg (Twist): Message de commande reçu
        """
        self.msg_q.append(msg)

def main(args=None):
    """Point d'entrée principal"""
    rclpy.init(args=args)
    serial_server = TwistIF("twist_interface")
    serial_server.loop()

if __name__ == '__main__':
    main()
