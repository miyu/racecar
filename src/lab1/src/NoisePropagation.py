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


def column(matrix, i):
    return [row[i] for row in matrix]

class NoisePropagation():
    def __init__(self, particles, state_lock):
        print("Initalized")
        self.odom = OdometryMotionModel(particles, state_lock)
        self.kine = KinematicMotionModel(particles, state_lock)
        self.odomMessage = []
        self.kineMessage = []
        #self.odomParticles = [particles]
        self.odomParticles = np.array([particles])
        self.kineParticles = [particles]

    def addOdomMessage(self,msg):
        #self.odomMessage.append(msg)
        self.odom.motion_cb(msg)
        #self.odomParticles.append(np.copy(self.odom.inner.particles))
        self.odomParticles = np.concatenate((self.odomParticles, [np.copy(self.odom.inner.particles)]), axis=0)

    def addKineVelMessage(self, msg):
        #self.kindVelMessage.append(msg)
        self.kine.motion_cb(msg)
        self.kineParticles.append(self.kine.inner.particles)

    def addKineServMessage(self, msg):
        #self.kindServMessage.append(msg)
        self.kine.servo_cb(msg)


    def plotOdomParticles(self):
        x_total = []
        y_total = []
        c = iter(cm.rainbow(np.linspace(0, 1, len(self.odomParticles)+1)))
        #assert len(self.odomParticles) == 20
        for i in range(len(self.odomParticles)):
            current_particle = self.odomParticles[i]
            #print(current_particle.shape)
            x = current_particle[:,0]
            #x = column(current_particle, 0)
            if x_total == []:
                x_total = x
            else:
                x_total += x
            print("X: ", x)
            assert len(x) == 500
            y = current_particle[:,1]
            #y = column(current_particle, 1)
            if y_total == []:
                y_total = y
            else:
                y_total += y
            #print("X", x)
            #plt.scatter(x, y)
            plt.scatter(x, y, color=next(c))
        plt.show()


if __name__=='__main__':
    rospy.init_node("open_loop_rollout", anonymous=True)
    BAG_FILENAME = str(rospy.get_param("~bag_file"))
    bag = rosbag.Bag(BAG_FILENAME)
    message = None
    initialize = True

    state_lock = Lock()
    particles = None

    noiseProp = None

    odomControlCount = 0
    kineControlVelCount = 0
    kineControlServCount = 0
    stepSize = 20
    try:
        for topic, msg, t in bag.read_messages():
            if topic == '/vesc/odom' and noiseProp is not None and odomControlCount < stepSize:
                print("time: ", msg.header.stamp.to_sec())
                odomControlCount += 1
                noiseProp.addOdomMessage(msg)
                print("odom")

            elif topic == '/initialpose':
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                theta = Utils.quaternion_to_angle(msg.pose.pose.orientation)
                particles = np.repeat(np.array([[x,y,theta]], dtype=np.float64),500,axis=0)
                print("Particles: ", particles)
                noiseProp = NoisePropagation(particles, state_lock)
                print("initalpose")
            elif topic == "/vesc/sensors/core" and noiseProp is not None and kineControlVelCount < 20:
                kineControlVelCount += 1
                noiseProp.addKineVelMessage(msg)
                print(topic)
            elif topic == "/vesc/sensors/servo_position_command" and noiseProp is not None and kineControlServCount < 20:
                kineControlServCount += 1
                noiseProp.addKineServMessage(msg)
                print(topic)
    finally:
        bag.close()
    #noiseProp.applyOdomModel()
    noiseProp.plotOdomParticles()
