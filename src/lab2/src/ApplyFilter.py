#!/usr/bin/env python

import rospy
import numpy as np
import csv
from convolve import convolve
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class Filter:
	def __init__(self, filter_path, sub_topic, pub_topic):
		# Load the filter from csv
		# Create the publisher and subscriber
		# Create a CvBridge object for converting sensor_msgs/Image into numpy arrays (and vice-versa)
		#		http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
		# Use the 'self' parameter to initialize as necessary
		
		print("Subscribing to", sub_topic)
		print("Publishing to", pub_topic)

		self.pub = rospy.Publisher(pub_topic, Image, queue_size=10)		
		self.sub = rospy.Subscriber(sub_topic, Image, self.apply_filter_cb)
		self.bridge = CvBridge()
		self.filter = np.array(list(csv.reader(open(filter_path, "rb"), delimiter=","))).astype("float")
		
	def apply_filter_cb(self, msg):
		# Apply the filter to the incoming image and publish the result
		# If the image has multiple channels, apply the filter to each channel independent of the other channels
		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
		except CvBridgeError as e:
			print(e)

		print ("original shape: ", cv_image.shape)
	
		a = cv_image.transpose(2,0,1) # to color channel, y, x
		b = self.filter
		print ("Shape:", a.shape, b.shape)
		
		output_height = a.shape[1] - b.shape[0] + 1
		output_width = a.shape[2] - b.shape[1] + 1
		output_array = np.zeros((a.shape[0], output_height, output_width), dtype=np.uint8)
		
		#fast convolve
		for channel in range(a.shape[0]):
			a_channel = a[channel]
			output_array[channel] = convolve(a_channel,b[np.newaxis,:,:], mode='valid')
		
		new_cv_image = output_array.transpose(1,2,0)
		print("Published shape:", new_cv_image.shape)
		
		print("Publishing:")
		try:
			self.pub.publish(self.bridge.cv2_to_imgmsg(new_cv_image, encoding="rgb8"))
		except CvBridgeError as e:
			print(e)
		print("DONE!")

		
if __name__ == '__main__':
	filter_path = None #The path to the csv file containing the filter
	sub_topic = None # The image topic to be subscribed to
	pub_topic = None # The topic to publish filtered images to

	rospy.init_node('apply_filter', anonymous=True)
	
	# Populate params with values passed by launch file
	filter_path = rospy.get_param('~filter_path')
	sub_topic = rospy.get_param('~sub_topic')	
	pub_topic = rospy.get_param('~pub_topic')
	
	print(filter_path, sub_topic, pub_topic)

	# Create a Filter object and pass it the loaded parameters
	filter = Filter(filter_path, sub_topic, pub_topic)
	
	rospy.spin()