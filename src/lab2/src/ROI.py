#!/usr/bin/env python
import rospy
import numpy as np
import csv
import cv2
from scipy import signal
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ROI:
    def __init__(self, sub_topic, pub_topic):
        print("Subscribing to", sub_topic)
        print("Publishing to", pub_topic)

        self.pub = rospy.Publisher(pub_topic, Image, queue_size=10)
        self.sub = rospy.Subscriber(sub_topic, Image, self.apply_filter_cb)
        self.bridge = CvBridge()
    def apply_filter_cb(self, msg):
        cv_image = None
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        except CvBridgeError as e:
            print(e)


        cv_image = cv2.GaussianBlur(cv_image, (5,5), 0)
        hsv  = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)

        #lower_blue = np.array([0, 150 ,  180 ])
        #upper_blue = np.array([255,255, 255])

        lower_blue = np.array([0, 60 ,  140 ])
        upper_blue = np.array([150,255, 180])


        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        res = cv2.bitwise_and(cv_image, cv_image, mask = mask)
        print("Publishing:")
        try:
            self.pub.publish(self.bridge.cv2_to_imgmsg(res, 'rgb8'))
        except CvBridgeError as e:
            print(e)
        print("DONE!")


if __name__ == '__main__':
	sub_topic = None # The image topic to be subscribed to
	pub_topic = None # The topic to publish filtered images to
	fast_convolve = False # Whether or not the nested for loop or Scipy's convolve method should be used for applying the filter

	rospy.init_node('apply_filter', anonymous=True)

	# Populate params with values passed by launch file
	sub_topic = rospy.get_param('~sub_topic')
	pub_topic = rospy.get_param('~pub_topic')


	# Create a Filter object and pass it the loaded parameters
	roi = ROI(sub_topic, pub_topic)

	rospy.spin()
