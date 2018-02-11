#!/usr/bin/env python
import numpy as np
import cv2
import time

"""
 Publish a filtered image from the given image data.
 Currently, it filters out all except light blue colors.

"""
class ROI:
    def __init__(self):
        self.error = 0.0

    def show(self, img):
        cv2.imshow('stuff', img)
        k = cv2.waitKey()
        if k == 27:
            return

    """
    A callback to remove all except some range of colors.
    """
    def apply_filter_cb(self):
        #cv_image = cv2.imread('arc.jpg', 1)
        cv_image = cv2.imread('arc.jpg', 1)
        assert cv_image is not None

        self.show(cv_image)
        cv_image = cv2.GaussianBlur(cv_image, (3,3), 0)
        hsv  = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        print(hsv)
        # To see the light blue tape
        # upper = np.array([150,255, 255])
        # lower = np.array([70, 100, 100 ])

        # To see the red tape

        #lower_l = np.array([0, 100, 100])
        #lower_h = np.array([20, 255, 255])
        higher_l = np.array([170, 100, 100])
        higher_r = np.array([179, 255, 255])
        #mask_rl = cv2.inRange(hsv, lower_l, lower_h)
        mask = cv2.inRange(hsv, higher_l, higher_r)


        #mask = cv2.inRange(hsv, lower, upper)
        print("mask " , mask.shape)
        res = cv2.bitwise_and(cv_image, cv_image, mask = mask)
        print(res)
        print("res ", res.shape)
        # res[:,:,0] = 0
        # res[:,:,1] = 0

        self.drawROI(res)

    def getError(self):
        return self.error

    """
        Draw out the ROI, and get the center
    """
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

        # Used to set the lines of the ROI
        bottomLineHeight = int(height - 10)
        topLineHeight = int(height - height/6)

        # Mask img for ROI
        mask = np.zeros((height, width), dtype = 'uint8')
        mask[(topLineHeight):(bottomLineHeight) ,:] = 1
        ROIimg = cv2.bitwise_and(img , img , mask = mask)

        gray = cv2.cvtColor(ROIimg, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        M = cv2.moments(cnts)

        # If the tape is not within the ROI, then don't calculate moment
        if M["m00"] == 0:
            print("Not on top of tape")
            self.error = 0.0
        else:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
            self.error = float(width/2.0 - float(cX))
            print('Cx: ', float(cX))
            print('Center:', width/2.0)

        cv2.line(img, (0,topLineHeight), (width,topLineHeight), color = (0, 255, 0))
        cv2.line(img, (0,bottomLineHeight), (width, bottomLineHeight), color = (0, 255, 0))

        self.show(img)

if __name__ == '__main__':
    roi = ROI()
    roi.apply_filter_cb()
