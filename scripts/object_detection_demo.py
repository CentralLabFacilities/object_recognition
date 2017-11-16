import numpy as np
import os
import sys
import cv2

from PIL import Image


from tf_detector import TfDetector



tf_detector = None

# helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def visualize_bounding_boxes(image_np, boxes, scores, classes):
    #visualization of detection results
    image_np = cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)

    #display image with bounding boxes using opencv
    height, width, channels = image_np.shape
    boxes = boxes[0]
    l = len(boxes)
    for i in range(0,l):
        bbox = boxes[i]
        prob = scores[0][i]
        if (prob > detection_threshold):
            cv2.rectangle(image_np, (int(bbox[1]*width), int(bbox[0]*height)), (int(bbox[3]*width), int(bbox[2]*height)), (0, 100, 200), 3)
            label = '%s %f' % (tf_detector.getLabel(classes[0][i]), prob)
            cv2.putText(image_np,label,(int(bbox[1]*width),int(bbox[0]*height)),0,0.7,(0,0,255),2)
    cv2.imshow('image_np',image_np)
    cv2.waitKey(0)


def detect_test_images(test_images_dir,detection_threshold):
    
    # Size, in inches, of the output images.
    image_size = (12, 8)
    
    for dirname, dirnames, filenames in os.walk(test_images_dir):
        for filename in filenames:
            image_path = dirname + filename
            print image_path

            #imgage = cv2.imread(image_path,0)
            image = Image.open(image_path)
        
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)

            (boxes, scores, classes) = tf_detector.detect(image_np, detection_threshold)

            visualize_bounding_boxes(image_np, boxes, scores, classes)
    cv2.destroyAllWindows()



if __name__ == "__main__":

	if len(sys.argv) < 5:
		print("usage: python object_detection_demo.py <path_to_graph> <path_to_labels> <path_to_test_images> <num_classes> <threshold>")
		exit(0)

	#get variables	
	detection_threshold = float(sys.argv[5])
	numClasses = int(sys.argv[4])
	pathToCkpt = sys.argv[1]
	pathToLabels = sys.argv[2]
	test_images_dir = sys.argv[3]

        tf_detector = TfDetector()
        print "load graph"
        tf_detector.load_graph(pathToCkpt, pathToLabels, numClasses)
        print "start detection on test images"
        detect_test_images(test_images_dir,detection_threshold)
        print "done"
