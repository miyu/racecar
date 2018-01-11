#!/usr/bin/env python

import rospy
import rosbag
from ackermann_msgs.msg import AckermannDriveStamped
import time
BAG_TOPIC = '/vesc/low_level/ackermann_cmd_mux/input/teleop'
PUB_TOPIC = '/vesc/high_level/ackermann_cmd_mux/input/nav_0'
PUB_RATE = 20

def follow_bag(bag_path, follow_backwards=False):
	pub = rospy.Publisher(PUB_TOPIC, AckermannDriveStamped, queue_size=10)
	r = rospy.Rate(PUB_RATE)
	bag = rosbag.Bag(bag_path)

	tuples = [x for x in bag.read_messages(topics=[BAG_TOPIC])]
	if follow_backwards:
		tuples.reverse()

	print("Begin playback")

	for topic, msg, t in tuples:
		if follow_backwards:
			msg.drive.speed = -1.0 * msg.drive.speed

		pub.publish(msg)
		r.sleep()

	bag.close()
	

if __name__ == '__main__':
	bag_path = None # The file path to the bag file
	follow_backwards = False # Whether or not the path should be followed backwards
	
	rospy.init_node('bag_follower', anonymous=True)
	
	# Populate param(s) with value(s) passed by launch file
	bag_path = rospy.get_param('~bag_path')
	follow_backwards = rospy.get_param('~follow_backwards')
	
	
	follow_bag(bag_path, follow_backwards)
