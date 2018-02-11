#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from PID import PID

if __name__=="__main__":
    red = np.uint8([[[0,0,255]]])
    hsv_red  = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
    print(hsv_red)
