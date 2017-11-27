import rospy
import time
import os
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node('VideoPublisher', anonymous=True)

VideoRaw = rospy.Publisher('/xtion/rgb/image_raw', Image, queue_size=1)

for filename in os.listdir('/media/sarah/media/test_images'):
        img = cv2.imread(os.path.join('/media/sarah/media/test_images',filename))
	img_msg = CvBridge().cv2_to_imgmsg(img, "rgb8")
	#cv2.imshow('img',img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	VideoRaw.publish(img_msg)
	time.sleep(0.5)
print "done"
