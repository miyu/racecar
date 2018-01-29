#!/usr/bin/env python

import rospy
import numpy as np
from tf.transformations import euler_from_quaternion

# THESE FUNCTIONS MAY OR MAY NOT BE HELPFUL IN YOUR IMPLEMENTATION
# IMPLEMENT/USE AS YOU PLEASE

def angle_to_quaternion(angle):
    pass

def quaternion_to_angle(q):
    return euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

def map_to_world(poses,map_info):
    pass

def world_to_map(poses, map_info):
    pass
