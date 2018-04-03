import rospy
import time
import os
from sensor_msgs.msg import Image
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node('VideoPublisher', anonymous=True)

VideoRaw = rospy.Publisher('/pepper_robot/sink/front/image_raw', Image, queue_size=1)
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    img_msg = CvBridge().cv2_to_imgmsg(frame, "bgr8")
    VideoRaw.publish(img_msg)
 #   time.sleep(0.1)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
