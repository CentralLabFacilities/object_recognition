#!/usr/bin/env python

from object_tracking_msgs.srv import Recognize
from object_tracking_msgs.msg import Recognition, CategoryProbability, Hypothesis, ObjectHypothesis
from object_tracking_msgs.srv import Classify
import rospy
import cv2
import random
import string
from cv_bridge import CvBridge, CvBridgeError


class segmenation_classification_bridge:

    def __init__(self):
        print("ros initialized!")
        self.service = rospy.Service('classify', Classify, self.classification_bridge)
        print("ros service Server startet.")
        self.classify = rospy.ServiceProxy('recognize', Recognize)
        print("created ros service proxy")

	if __debug__:
    		print "In debug mode"
	else:
    		print "Not in debug mode"


    def classification_bridge(self, req):
        print("got request with " + str(len(req.objects)) + " cropped images")

        response = []
        for cropped_image in req.objects:
            labels = []
            
            result = self.classify(cropped_image)
	    
            for r in result.recognitions:	
		objects = (r.categorical_distribution.probabilities)
		for o in objects:
			label_prob = Hypothesis(o.label, o.probability)
			labels.append(label_prob)
			

            response.append(ObjectHypothesis(labels))
	
	if __debug__:
		print "Visualization"
		

		#bridge = CvBridge()
		#for im in req.objects:
			#cv_image = bridge.imgmsg_to_cv2(im, desired_encoding="rgb8")
			#cv2.namedWindow('test')
			#cv2.imshow('test', cv_image)		
			#cv2.imshow(''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10)),cv_image)
			#cv2.waitKey(0)
			#cv2.destroyWindow('test')
		#cv2.destroyAllWindows()

		#image_message_1 = req.objects[0]
		#image_message_2 = req.objects[1]
		#bridge = CvBridge()
		#cv_image_1 = bridge.imgmsg_to_cv2(image_message_1, desired_encoding="bgr8")
		#cv_image_2 = bridge.imgmsg_to_cv2(image_message_2, desired_encoding="bgr8")
		
		#print "so far"
		
		#cv2.imshow('image1',cv_image_1)
		#cv2.waitKey(3)
		#cv2.imshow('image1',cv_image_2)
		#cv2.waitKey(3)
		#cv2.destroyAllWindows()

		#print "done"    		
	


        return {"hypotheses":response}

if __name__ == "__main__":
    rospy.init_node('segmentation_classification_bridge')
    seg = segmenation_classification_bridge()
    rospy.spin()
