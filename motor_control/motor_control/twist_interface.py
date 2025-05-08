#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import serial
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from struct import unpack
from numpy import sin, cos, degrees, radians, pi, clip
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import Quaternion

# ===============Command Bits Reference===========================
# const uint8_t m_dir = 1<<0; // Rotation direction 0:backward 1:forward
# const uint8_t m_free = 1<<1; // Free wheel 0:locked 1:freewheel
# const uint8_t m_right = 1<<2; // Apply to right motor 0:no 1:yes
# const uint8_t m_left = 1<<3; // Apply to left motor 0:no 1:yes
# const uint8_t m_rst = 1<<7; // Reset encoder counters
# ================================================================

TICKS_REV = 3840
TICKS_HALF_REV = TICKS_REV / 2
WHEEL_RADIUS = 0.032
WHEELS_SEP = 0.225
CONTROL_LOOP_PERIOD = 10

class TwistIF(Node):
    new_cmd = False
    l_speed = 0
    r_speed = 0
    l_odom = 0
    r_odom = 0
    last_l_odom = 0
    last_r_odom = 0
    last_now = 0
    x_odom = 0
    y_odom = 0
    a_odom = 0
    msg_q = []
    msg_odom = Odometry()

    def __init__(self, node_name:str='twist_interface'):
        super().__init__(node_name)
        #Default Value declarations of ros2 params:
        self.declare_parameters(
        namespace='',
        parameters=[
            ('device', '/dev/ttyACM0'), #device we are trasmitting to & receiving messages from
            ('wheel_instructions_topic', 'wheel_instructions_topic'),
        ]
        )
        self.wheel_topic_name = self.get_param_str('wheel_instructions_topic')
        self.device_name = self.get_param_str('device')
        self.ser = serial.Serial(self.device_name,
                           115200, #Note: Baud Rate must be the same in the arduino program, otherwise signal is not received!
                           timeout=.1)
        
        self.subscriber = self.create_subscription(Twist, 
                                              self.wheel_topic_name, 
                                              self.twist_listener_callback, 
                                              10)
        self.publisher = self.create_publisher(Odometry, 'robot_odom', 10)
        self.subscriber # prevent unused variable warning
        self.ser.reset_input_buffer()
        
        self.msg_odom.pose.pose.position.x = 0.0
        self.msg_odom.pose.pose.position.y = 0.0
        self.msg_odom.pose.pose.position.z = 0.0
        self.msg_odom.pose.pose.orientation.x = 0.0
        self.msg_odom.pose.pose.orientation.y = 0.0
        self.msg_odom.pose.pose.orientation.z = 0.0
        self.msg_odom.pose.pose.orientation.w = 1.0
        self.msg_odom.twist.twist.linear.x = 0.0
        self.msg_odom.twist.twist.linear.y = 0.0
        self.msg_odom.twist.twist.linear.z = 0.0
        self.msg_odom.twist.twist.angular.x = 0.0
        self.msg_odom.twist.twist.angular.y = 0.0
        self.msg_odom.twist.twist.angular.z = 0.0

        self.get_logger().info("succesfully started!")

    def serial_read_bytes(self, length=8):
        sermsg = self.ser.readline()
        while len(sermsg) <= length:
            sermsg += self.ser.readline()
        self.l_odom = unpack('<l', sermsg[0:4])[0]
        self.r_odom = unpack('<l', sermsg[4:8])[0]

    def publish_odom(self):
        l_ticks = self.l_odom - self.last_l_odom
        r_ticks = self.r_odom - self.last_r_odom

        l_dist = WHEEL_RADIUS * 2 * pi * l_ticks / TICKS_REV
        r_dist = WHEEL_RADIUS * 2 * pi * r_ticks / TICKS_REV
        avg_dist = (l_dist + r_dist) / 2
        deltaA = (l_dist - r_dist) / WHEELS_SEP
        self.a_odom += deltaA

        deltaX = cos(self.a_odom) * avg_dist
        deltaY = sin(self.a_odom) * avg_dist
        self.x_odom += deltaX
        self.y_odom += deltaY

        now = self.get_clock().now()
        interval = (now.nanoseconds / 1000000000) - self.last_now

        x_vel = deltaX / interval
        y_vel = deltaY / interval
        a_vel = deltaA / interval

        self.msg_odom.header.stamp = now.to_msg()
        self.msg_odom.pose.pose.position.x = self.x_odom
        self.msg_odom.pose.pose.position.y = self.y_odom
        q = quaternion_from_euler(0, 0, self.a_odom)
        self.msg_odom.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.msg_odom.twist.twist.linear.x = x_vel
        self.msg_odom.twist.twist.linear.y = y_vel
        self.msg_odom.twist.twist.angular.z = a_vel
        self.publisher.publish(self.msg_odom)

        self.last_l_odom = self.l_odom
        self.last_r_odom = self.r_odom
        self.last_now = now.nanoseconds / 1000000000

    def twist_translator(self, msg):
        avgVel = msg.linear.x
        deltaVel = msg.angular.z * WHEELS_SEP / 2
        dl = avgVel + deltaVel
        dr = avgVel - deltaVel

        l_ticks = TICKS_REV * dl / (WHEEL_RADIUS * 2 * pi) * CONTROL_LOOP_PERIOD / 1000
        r_ticks = TICKS_REV * dr / (WHEEL_RADIUS * 2 * pi) * CONTROL_LOOP_PERIOD / 1000

        self.l_speed = clip(int(l_ticks), -255, 255)
        self.r_speed = clip(int(r_ticks), -255, 255)

    def send_serial_cmd(self):
        if self.l_speed == self.r_speed:
            bits, speed = self.cmd_bits_abs_speed(self.l_speed)
            self.ser.write(bytes([0b00001100 | bits, speed]))
        else:
            bits, speed = self.cmd_bits_abs_speed(self.l_speed)
            self.ser.write(bytes([0b00001000 | bits, speed]))
            bits, speed = self.cmd_bits_abs_speed(self.r_speed)
            self.ser.write(bytes([0b00000100 | bits, speed]))

    def cmd_bits_abs_speed(self, speed):
        return 0b00000000 if (speed < 0) else (0b00000011 if (speed == 0) else 0b00000001), abs(speed)
        # return 0b00000001 if (speed < 0) else (0b00000010 if (speed == 0) else 0b00000000), abs(speed)

    def loop(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)
            self.serial_read_bytes(8)
            self.publish_odom()
            if self.msg_q.__len__():
                self.twist_translator(self.msg_q.pop(0))
                self.send_serial_cmd()
                if self.msg_q.__len__() > 9:
                    self.msg_q.clear()
                    print(f'Purged MSG queue (len {self.msg_q.__len__()})')

    def get_param_float(self, name):
        try:
            return float(self.get_parameter(name).get_parameter_value().double_value)
        except:
            pass

    def get_param_str(self, name):
        try:
            return self.get_parameter(name).get_parameter_value().string_value
        except:
            pass

    def twist_listener_callback(self, msg):
        # self.new_cmd = True
        # self.msg_q = msg
        self.msg_q.append(msg)


def main(args=None):
    rclpy.init(args=args)
    serial_server = TwistIF("twist_interface")
    serial_server.loop()
  

if __name__ == '__main__':
    main()
