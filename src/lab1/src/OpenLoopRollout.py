#!/usr/bin/env python

import rospy
import rosbag
import time
import numpy as np
import matplotlib.pyplot as plt
from threading import Lock
import utils as Utils

import matplotlib.colors as colors
import matplotlib.cm as cm
from std_msgs.msg import Int32, String
from MotionModel import OdometryMotionModel
from MotionModel import KinematicMotionModel
from nav_msgs.srv import GetMap



class OpenLoopRollout():
    def __init__(self, particles, state_lock):
        #print("Initalized")
        self.particle_list= np.array(particles)
        self.particle_list_kin = np.array(particles)
        self.true_pose_list = None
        self.motionModel = OdometryMotionModel(particles, state_lock)
        self.kinematic = KinematicMotionModel(particles, state_lock)

    def apply_motion_model(self, msg):
        #print(msg)
        self.motionModel.motion_cb(msg)
        #print("Motion model particle: ",self.motionModel.inner.particles)
        self.particle_list = np.concatenate((self.particle_list, self.motionModel.inner.particles))

    def apply_kin_model(self, msg, topic):
        if topic == "/vesc/sensors/core":
            self.kinematic.motion_cb(msg)
            self.particle_list_kin = np.concatenate((self.particle_list_kin, self.kinematic.inner.particles))
        elif topic == "/vesc/sensors/servo_position_command":
            self.kinematic.servo_cb(msg)

    def add_true_pose(self, particle):
        if self.true_pose_list is None:
            self.true_pose_list = np.array(particle)
        else:
            self.true_pose_list = np.concatenate((self.true_pose_list, particle))


    def plot_particles(self):
        #print("First Particle: " ,self.particle_list)
        x = self.particle_list[:,0]
        y = self.particle_list[:,1]

        x_kin = self.particle_list_kin[:,0]
        y_kin = self.particle_list_kin[:,1]

        x_true = self.true_pose_list[:,0]
        y_true = self.true_pose_list[:,1]
        #print("X: ", x)

        #print("Y: ", y)

        #plt.plot(x,y)
        plt.plot(x_kin, y_kin)
        plt.plot(x_true, y_true)
        plt.show()

if __name__=='__main__':
    rospy.init_node("open_loop_rollout", anonymous=True)
    BAG_FILENAME = str(rospy.get_param("~bag_file"))
    bag = rosbag.Bag(BAG_FILENAME)
    message = None
    initialize = True

    state_lock = Lock()
    particles = None

    oLR = None

    try:
        for topic, msg, t in bag.read_messages():
            #break
            # See if it's a map message
            #if hasattr(msg, 'origin'):

                #print(msg)

            if topic == '/vesc/odom' and oLR is not None:
            #if hasattr(msg, 'pose'):
                # See if it's a odometry reading
                #if hasattr(msg.pose, 'pose'):
                """
                if initialize:
                    x = msg.pose.pose.position.x
                    y = msg.pose.pose.position.y
                    theta = Utils.quaternion_to_angle(msg.pose.pose.orientation)
                    particles = np.array([[x,y,theta]], dtype=np.float64)
                    oLR = OpenLoopRollout(particles, state_lock)
                    initialize = False
                else:
                """
                oLR.apply_motion_model(msg)
            elif topic == '/initialpose':
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                theta = Utils.quaternion_to_angle(msg.pose.pose.orientation)
                particles = np.array([[x,y,theta]], dtype=np.float64)
                oLR = OpenLoopRollout(particles, state_lock)
                #initialize = False
            elif 'map' in topic:
                #print(msg)
                print('')
            elif topic == '/pf/ta/viz/inferred_pose' and oLR is not None:
                x = msg.pose.position.x
                y = msg.pose.position.y

                oLR.add_true_pose(np.array([[x, y]]))
            elif (topic == "/vesc/sensors/core" or topic == "/vesc/sensors/servo_position_command") and oLR is not None:
                oLR.apply_kin_model(msg, topic)
            elif 'offset' in topic:
                print(topic)

    finally:
        bag.close()
    #oLR = OpenLoopRollout(message)

    oLR.plot_particles()
