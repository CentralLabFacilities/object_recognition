#!/usr/bin/env python

import sys
import rospy
import os

import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import roslib
import time

from object_tracking_msgs.srv import Classify

def art_call_client():
    rospy.wait_for_service('classify')
    try:
	
	req = []
	images_object = []
	bridge = CvBridge()
	
    	for filename in os.listdir('/home/susan/object_recognition/white_up/tensorset/apple'):
        	img = cv2.imread(os.path.join('/home/susan/object_recognition/white_up/tensorset/apple',filename))
        	if img is not None:
            		req.append(bridge.cv2_to_imgmsg(img, encoding="bgr8"))

	
	

        art_call = rospy.ServiceProxy('classify', Classify)
	print "created ros service proxy - classify"
	
	"""
	cv_image_1 = cv2.imread('/home/susan/object_recognition/white_up/tensorset/apple/apple-2017-06-26-17-09-41_788859.jpg',1)
	cv_image_2 = cv2.imread('/home/susan/object_recognition/white_up/tensorset/peas/peas-2017-06-29-12-52-13_502532.jpg',1)
	image_message_1 = bridge.cv2_to_imgmsg(cv_image_1, encoding="bgr8")
	image_message_2 = bridge.cv2_to_imgmsg(cv_image_2, encoding="bgr8")
	req.append(image_message_1)
	req.append(image_message_2)
	"""

	#print "%s" %art_call(req)
	
	for object in art_call(req).hypotheses:
		print "%s" %object.hypotheses[0].label
	print "called classify with image_message"

    except rospy.ServiceException, e:
        print "Service call failed: %s"%e 


if __name__ == "__main__":
    art_call_client()
