#!/usr/bin/env python

import rospy
import numpy as np
import csv
from scipy import signal
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class Filter:
	def __init__(self, filter_path, sub_topic, pub_topic, fast_convolve=False):
		# Load the filter from csv
		# Create the publisher and subscriber
		# Create a CvBridge object for converting sensor_msgs/Image into numpy arrays (and vice-versa)
		#		http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
		# Use the 'self' parameter to initialize as necessary
		
		self.pub = rospy.Publisher(pub_topic, Image, queue_size=10)		
		self.sub = rospy.Subscriber(sub_topic, Image, apply_filter_cb)
		self.bridge = CvBridge()
		self.filter = numpy.array(list(csv.reader(open(filter_path, "rb"), delimiter=","))).astype("float")
		self.fast_convolve = fast_convolve
		

	def apply_filter_cb(self, msg):
		# Apply the filter to the incoming image and publish the result
		# If the image has multiple channels, apply the filter to each channel independent of the other channels
		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
		except CvBridgeError as e:
			print(e)
		
		if self.fast_convolve:
			#fast - scipy convolve
		else:
			#slow - loop

		try:
			self.pub.publish(self.bridge.cv2_to_imgmsg(cv_image, encoding="passthrough"))
		except CvBridgeError as e:
			print(e)

		
if __name__ == '__main__':
	filter_path = None #The path to the csv file containing the filter
	sub_topic = None # The image topic to be subscribed to
	pub_topic = None # The topic to publish filtered images to
	fast_convolve = False # Whether or not the nested for loop or Scipy's convolve method should be used for applying the filter

	rospy.init_node('apply_filter', anonymous=True)
	
	# Populate params with values passed by launch file
	filter_path = rospy.get_param('~filter_path')
	sub_topic = rospy.get_param('~sub_topic')	
	pub_topic = rospy.get_param('~pub_topic')
	fast_convolve = rospy.get_param('~fast_convolve')
	
	
	# Create a Filter object and pass it the loaded parameters
	filter = Filter(filter_path, sub_topic, pub_topic, fast_convolve)
	
	rospy.spin()
	
