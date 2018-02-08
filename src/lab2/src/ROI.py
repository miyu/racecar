#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


"""
 Publish a filtered image from the given image data.
 Currently, it filters out all except light blue colors.

 sub_topic: where the original image is from
 pub_topic: where to publish the new image to

"""
class ROI:
    def __init__(self, sub_topic, pub_topic):
        print("Subscribing to", sub_topic)
        print("Publishing to", pub_topic)

        self.pub = rospy.Publisher(pub_topic, Image, queue_size=10)
        self.sub = rospy.Subscriber(sub_topic, Image, self.apply_filter_cb)
        self.bridge = CvBridge()

    """
    A callback to remove all except some range of colors.

    Assuming that msg is sensor_msgs/Image
    """
    def apply_filter_cb(self, msg):
        cv_image = None
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        except CvBridgeError as e:
            print(e)


        cv_image = cv2.GaussianBlur(cv_image, (3,3), 0)

        hsv  = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)

        # To see the light blue
        lower = np.array([35, 60 ,  100 ])
        upper = np.array([150,255, 255])

        # To see the red tape

        mask = cv2.inRange(hsv, lower, upper)

        res = cv2.bitwise_and(cv_image, cv_image, mask = mask)

        print("Publishing:")
        try:
            res[:,:,0] = 0
            res[:,:,1] = 0


            ROI = self.drawROI(res)

            #edges = self.getEdges(ROI)


            self.pub.publish(self.bridge.cv2_to_imgmsg(res, 'rgb8'))
            #self.pub.publish(self.bridge.cv2_to_imgmsg(edges))
            #self.pub.publish(self.bridge.cv2_to_imgmsg(ROI))
        except CvBridgeError as e:
            print(e)
        print("DONE!")


    """
        Get the edges of the image. Image is of type 8UC1
    """
    def getEdges(self, img):
        edges = cv2.Canny(img, 150, 200)
        return edges

    def drawROI(self, img):
        height = None
        width = None
        channels = 1

        edges = None


        if len(img.shape) == 2:
            height, width = img.shape
            #edges = img.copy()
        else:
            height, width, channels  = img.shape

            #edges = self.getEdges(img)

        # Used to set the line of the ROI
        bottomLineHeight = 10
        topLineHeight = height/6

        mask = np.zeros((height, width), dtype = 'uint8')

        mask[(height - topLineHeight):(height - bottomLineHeight) ,:] = 1

        ROIimg = cv2.bitwise_and(img ,img , mask = mask)

        gray = cv2.cvtColor(ROIimg, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    	M = cv2.moments(cnts)

        if M["m00"] == 0:
            print("Not on top of tape")

        else:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
            cv2.line(img, (0,height-topLineHeight), (width,height-topLineHeight), color = (0, 255, 0))
            cv2.line(img, (0,height-bottomLineHeight), (width, height-bottomLineHeight), color = (0, 255, 0))
        return thresh

if __name__ == '__main__':
	sub_topic = None # The image topic to be subscribed to
	pub_topic = None # The topic to publish filtered images to
	fast_convolve = False # Whether or not the nested for loop or Scipy's convolve method should be used for applying the filter

	rospy.init_node('apply_filter', anonymous=True)

	# Populate params with values passed by launch file
	sub_topic = rospy.get_param('~sub_topic')
	pub_topic = rospy.get_param('~pub_topic')

	# Create a ROI object and pass it the loaded parameters
	roi = ROI(sub_topic, pub_topic)

	rospy.spin()
