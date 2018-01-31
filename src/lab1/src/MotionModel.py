#!/usr/bin/env python

import math
import rospy
import numpy as np
import utils as Utils
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from vesc_msgs.msg import VescStateStamped
from threading import Lock
from Debug import print_locks, print_benchmark
import time

from InternalMotionModel import InternalOdometryMotionModel, InternalKinematicMotionModel

def np_array_or(x, y):
    return x if x is not None else y


class OdometryMotionModel:
    def __init__(self, particles, state_lock=None):
        self.sub = rospy.Subscriber("/vesc/odom",Odometry, self.motion_cb )
        self.state_lock = state_lock or Lock()

        # init internal odometry model
        particles = np_array_or(particles, np.array([[0.0, 0.0, 0.0]], dtype=np.float64))
        self.inner = InternalOdometryMotionModel(particles, [0, 0, 0])

    def motion_cb(self, msg):
        print_locks("Entering lock motion_cb")
        self.state_lock.acquire()
        print_locks("Entered lock motion_cb")
        start_time = time.time()

        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        theta = Utils.quaternion_to_angle(msg.pose.pose.orientation)
        #print("have", x, y, theta)
        self.inner.update([x, y, theta])

        print_locks("Releasing lock motion_cb")
        print_benchmark("odometry motion_cb", start_time, time.time())
        self.state_lock.release()

class KinematicMotionModel:
    def __init__(self, particles=None, state_lock=None):
        # state tracking
        self.last_servo_cmd = None # The most recent servo command
        self.last_vesc_stamp = None # The time stamp from the previous vesc state msg
        self.state_lock = state_lock or Lock()

        # config
        self.SPEED_TO_ERPM_OFFSET = 0.0#float(rospy.get_param("/vesc/speed_to_erpm_offset")) # Offset conversion param from rpm to speed
        self.SPEED_TO_ERPM_GAIN   = 4614.0#float(rospy.get_param("/vesc/speed_to_erpm_gain"))   # Gain conversion param from rpm to speed
        self.STEERING_TO_SERVO_OFFSET = 0.5304#float(rospy.get_param("/vesc/steering_angle_to_servo_offset")) # Offset conversion param from servo position to steering angle
        self.STEERING_TO_SERVO_GAIN   = -1.2135#float(rospy.get_param("/vesc/steering_angle_to_servo_gain")) # Gain conversion param from servo position to steering angle

        # This subscriber just caches the most recent servo position command
        self.servo_pos_sub = rospy.Subscriber(rospy.get_param("~servo_pos_topic", "/vesc/sensors/servo_position_command"), Float64,self.servo_cb, queue_size=1)

        #To get velocity
        #self.motion_vel_sub = rospy.Subscriber(rospy.get_param("~vel", "/vesc/sensors/core"), VescStateStamped, self.motion_cb)

        # init internal kinematic model
        particles = np_array_or(particles, np.array([[0, 0, 0]], dtype=np.float64))
        self.inner = InternalKinematicMotionModel(particles)

    def servo_cb(self, msg):
        self.state_lock.acquire()
        self.last_servo_cmd = msg.data # Just update servo command
        self.state_lock.release()

    def motion_cb(self, msg):
        print_locks("Entering lock motion_cb")
        self.state_lock.acquire()
        print_locks("Entered lock motion_cb")
        start_time = time.time()

        # if no servo info, skip
        if self.last_servo_cmd is None:
            print("Releasing lock motion_cb early (no servo command)")
            self.state_lock.release()
            return

        # if no prev timestamp, then play as if dt is 0
        if self.last_vesc_stamp is None:
            self.last_vesc_stamp = msg.header.stamp

        dt = (msg.header.stamp - self.last_vesc_stamp).to_sec()
        self.last_vesc_stamp = msg.header.stamp

        # Convert raw msgs to controls
        # Note that control = (raw_msg_val - offset_param) / gain_param
        curr_speed = float(float(msg.state.speed) - float(self.SPEED_TO_ERPM_OFFSET)) / float(self.SPEED_TO_ERPM_GAIN)
        curr_steering_angle = float(float(self.last_servo_cmd) - float(self.STEERING_TO_SERVO_OFFSET)) / float(self.STEERING_TO_SERVO_GAIN)

        # print("Velocity ", curr_speed)
        # print("Steering Angle: ", curr_steering_angle, " -- ", self.last_servo_cmd, " -- ", self.STEERING_TO_SERVO_OFFSET)
        # print("Delta Time: ", dt)

        self.inner.update([curr_speed, curr_steering_angle, dt])

        print_locks("Releasing lock motion_cb")
        print_benchmark("kinematic motion_cb", start_time, time.time())
        self.state_lock.release()

if __name__ == '__main__':
    rospy.init_node('OdoTest1', anonymous=True)
    #Odo = OdometryMotionModel(None, None)
    print("Enter main")
    Kin = KinematicMotionModel(None, None)
    rospy.spin()
    pass
