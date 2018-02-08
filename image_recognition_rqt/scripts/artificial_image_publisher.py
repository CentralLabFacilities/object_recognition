import rospy
import time
import os
from sensor_msgs.msg import Image
import sys

import cv2
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node('VideoPublisher', anonymous=True)
VideoRaw = rospy.Publisher('/xtion/rgb/image_raw', Image, queue_size=1)

def publish(img):
	img_msg = CvBridge().cv2_to_imgmsg(img, "rgb8")
	VideoRaw.publish(img_msg)
	time.sleep(0.5)


if __name__ == "__main__":

    # check for correct argument size
    if not len(sys.argv) >= 2:
        print '\033[91m' + 'Argument Error!\nUsage: python artificila_image_publisher.py path_to_dataset' + '\033[0m'
        exit(1)
    # check if argument given is a directory
    if not os.path.isdir(sys.argv[1]):
        print '\033[91m' + sys.argv[1] + ' is not a directory!' + '\033[0m'
        exit(1)
    path = sys.argv[1]
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            imagepath = "{}/{}".format(dirname,filename)
            if (os.path.isfile(imagepath)):
                # read image
				image = cv2.imread(imagepath, 3)
				cv2.imshow('img',image)
				cv2.waitKey(0)
				publish(image)

	cv2.destroyAllWindows()
    print '\033[1m\033[92mDone!\033[0m'
