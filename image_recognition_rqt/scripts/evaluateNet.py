from image_recognition_util import evaluate
from image_recognition_util.object import Object
from image_recognition_util.object import BoundingBox
import rospy
import rostopic
import rosservice
import cv2
from image_recognition_util.objectset_utils import ObjectsetUtils
from cv_bridge import CvBridge, CvBridgeError
import sys
import os
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from object_tracking_msgs.msg import CategoryProbability


threshold = 0.2
srv_detect = rospy.ServiceProxy("/detect", rosservice.get_service_class_by_name("/detect"))
srv_recognize = rospy.ServiceProxy("/recognize", rosservice.get_service_class_by_name("/recognize"))
util = ObjectsetUtils()
bridge = CvBridge()
total_correct = 0
total_wrong = 0
total_images = 0
total_unkown_detected = 0
total_to_find = 0
index = 0

def readAnnotated(labelpath, label_map, num_classes):
    annotatedList = []
    # "label" is the id
    labelList = util.getLabelList(labelpath)
    bboxList = util.getRoiList(labelpath)

    # get label
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                use_display_name=True)
    for i in range(0,len(labelList)):
        label = labelList[i]
        xmin = bboxList[i].xmin
        xmax = bboxList[i].xmax
        ymin = bboxList[i].ymin
        ymax = bboxList[i].ymax
        category_index = label_map_util.create_category_index(categories)
        id = int(label) + 1
        class_text = category_index[id]['name']

        a = Object(class_text,1.0,xmin,xmax,ymin,ymax)
        annotatedList.append(a)
    return annotatedList

def detect(cvImage):
    try:
        result = srv_detect(image=bridge.cv2_to_imgmsg(cvImage, "bgr8"))
    except Exception as e:
        print("Service Exception", str(e))
        return
    #print("convert detections to objectList")
    detections = convertMsgToObject(result.detections)
    return detections

def convertMsgToObject(results):
    detections = []
    for res in results:
        d = Object(res.category_probability.label, res.category_probability.probability,
                   res.bbox.x_min, res.bbox.x_max, res.bbox.y_min, res.bbox.y_max)
        detections.append(d)
    return detections

def detectAndRecognize(cvImage):
    height, width, channels = cvImage.shape
    try:
        #detection
        result = srv_detect(image=bridge.cv2_to_imgmsg(cvImage, "bgr8"))
        #recognition for each object/roi
        detections = result.detections
        for d in detections:
            xmin = int(d.bbox.x_min*width)
            xmax = int(d.bbox.x_max*width)
            ymin = int(d.bbox.y_min*height)
            ymax = int(d.bbox.y_max*height)
            roi = cvImage[ymin:ymax, xmin:xmax]
            try:
                res = srv_recognize(image=bridge.cv2_to_imgmsg(roi, "bgr8"))
                # assume res always has len=1, is this correct?
                r = res.recognitions[0]
                best = CategoryProbability(label="unknown", probability=r.categorical_distribution.unknown_probability)
                # get label with highest probability
                for p in r.categorical_distribution.probabilities:
                    if p.probability > best.probability:
                        best = p
                # change label and probability
                #print("detected {}, {}".format(d.category_probability.label, d.category_probability.probability))
                #print("recognized {}, {}".format(best.label, best.probability))
                d.category_probability.label = best.label
                d.category_probability.probability = best.probability
                if (best.label=="unkown"):
                    print("unknown object found")
            except Exception as e:
                print("Service Expcetion during recognition", str(e))
                return
    except Exception as e:
        print("Service Exception during detection", str(e))
        return
    #print("convert detections to objectList")
    detections = convertMsgToObject(result.detections)
    return detections

def evaluateImage(labelpath, imagepath, label_map, num_classes):
    cvImage = cv2.imread(imagepath, 3)
    annotatedList = readAnnotated(labelpath, label_map, num_classes)
    to_find = len(annotatedList)
    detectedList = detectAndRecognize(cvImage)
    correct, wrong, unkown_detected, image = evaluate.evaluateDetection(annotatedList, detectedList, threshold, cvImage)
    return correct, wrong, unkown_detected, to_find, image



if __name__ == "__main__":

    # check for correct argument size
    if not len(sys.argv) >= 4:
        print '\033[91m' + 'Argument Error!\nUsage: python fix_txt.py path_to_dataset path_to_labelmap num_classes [save_image_path]' + '\033[0m'
        exit(1)
    # check if argument given is a directory
    if not os.path.isdir(sys.argv[1]):
        print '\033[91m' + sys.argv[1] + ' is not a directory!' + '\033[0m'
        exit(1)
    path = sys.argv[1]
    label_map = label_map_util.load_labelmap(sys.argv[2])
    num_classes = int(sys.argv[3])
    if (len(sys.argv) == 5):
        savepath = sys.argv[4]
    else:
        savepath = "/tmp"
    print("save eval images in {}".format(savepath))
    print("treshold: {}".format(threshold))
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            labelpath = dirname + '/' + filename
            if 'labels' in labelpath and '.txt' in labelpath:
                imagepath = "{}/images/{}.jpg".format(dirname[:-7],filename[:-4])

                if (os.path.isfile(imagepath)):
                    #print("evaluate ", imagepath)
                    correct, wrong, unkown_detected, to_find, image = evaluateImage(labelpath,imagepath, label_map, num_classes)
                    total_correct = total_correct + correct
                    total_wrong = total_wrong + wrong
                    total_unkown_detected = total_unkown_detected + unkown_detected
                    total_images = total_images + 1
                    total_to_find = total_to_find + to_find
                    save_filename = '{}/eval_image{}.jpg'.format(savepath, index)
                    #print("save image to: ", save_filename)
                    cv2.imwrite(save_filename, image)
                    index = index + 1

    print("detected ",total_correct, " correct and ", total_wrong, " wrong of ",total_to_find, " objects,")
    print("detected ",total_unkown_detected, " unknown objects in ",total_images," images")
    print '\033[1m\033[92mDone!\033[0m'

