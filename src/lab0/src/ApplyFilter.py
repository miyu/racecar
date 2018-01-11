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
		
		a = cv_image.transpose(2,0,1)
		b = self.filter
		
    	output_height = a.shape[1] - b.shape[0] + 1
    	output_width = a.shape[2] - b.shape[1] + 1
    	output_array = np.zeros((a.shape[0], output_height, output_width))
		
		if self.fast_convolve:
			#fast - scipy convolve
    		for channel in range(a.shape[0]):
    			a_channel = a[channel]
    			output_array[channel] = signal.convolve(a_channel,np.flipud(np.fliplr(b)), mode='valid')
    		#print('a: ' + str(a))
    		#print('b: ' + str(b))
    		print(output_array)
		else:
			#slow - loop
    		for channel in range(output_array.shape[0]):
    			for h in range(output_array.shape[1]):
    				for w in range(output_array.shape[2]):
    					leftbound = w
    					upperbound = h
    					rightbound = w + b.shape[1]
    					lowerbound = h + b.shape[0]
    					#print(leftbound, rightbound, upperbound, lowerbound)
    					#print(a[channel][upperbound:lowerbound])
    					slice = a[channel, upperbound:lowerbound, leftbound:rightbound]
    					print(slice.shape)
    					print(b.shape)
    					output_array[channel][h][w] = np.sum(np.multiply(slice, b))
    					
    		#print('a: ' + str(a))
    		#print('b: ' + str(b))
    		print(output_array)
		
		new_cv_image = output_array.transpose(1,2,0)
		
		try:
			self.pub.publish(self.bridge.cv2_to_imgmsg(new_cv_image, encoding="passthrough"))
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
	
