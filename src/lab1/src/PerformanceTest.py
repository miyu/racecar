#!/usr/bin/env python

import rospy
import rosbag
import time
import numpy as np
import matplotlib.pyplot as plt
from threading import Lock

import matplotlib.colors as colors
import matplotlib.cm as cm
from std_msgs.msg import Int32, String
from SensorModel import SensorModel
from nav_msgs.srv import GetMap


class PerformanceTest():
    def __init__(self):
        print("hello")

if __name__=='__main__':
    rospy.init_node("performance_test", anonymous=True)
    BAG_FILENAME = str(rospy.get_param("~bag_file"))
    bag = rosbag.Bag(BAG_FILENAME)
    message = None
    try:
        for topic, msg, t in bag.read_messages():
            message = msg
    finally:
        bag.close()
