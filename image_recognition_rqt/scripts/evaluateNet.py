# openCV
import cv2
from cv_bridge import CvBridge, CvBridgeError

# sys
import sys
import os

# object rec
from image_recognition_util import evaluate
from image_recognition_util.object import Object
from image_recognition_util.object import BoundingBox
from image_recognition_util.objectset_utils import ObjectsetUtils
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from object_tracking_msgs.msg import CategoryProbability
from tensorflow_ros import detector
from tensorflow_ros import recognizer


class EvaluateNet:
    def __init__(self, threshold=0.5):
        # variables
        self.threshold = threshold
        self.total_correct = 0
        self.total_wrong = 0
        self.total_images = 0
        self.total_unkown_detected = 0
        self.total_to_find = 0
        self.index = 0

        self._filename = "/tmp/rec_image.png"

        # objects
        self.detector = detector.Detector(detection_threshold=self.threshold)
        self.recognizer = recognizer.Recognizer()
        self.util = ObjectsetUtils()


    def readAnnotated(self, labelpath, label_map, num_classes):
        annotatedList = []
        # "label" is the id
        labelList = self.util.getLabelList(labelpath)
        bboxList = self.util.getRoiList(labelpath)

        # get label
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                    use_display_name=True)
        for i in range(0, len(labelList)):
            label = labelList[i]
            xmin = bboxList[i].xmin
            xmax = bboxList[i].xmax
            ymin = bboxList[i].ymin
            ymax = bboxList[i].ymax
            category_index = label_map_util.create_category_index(categories)
            id = int(label) + 1
            class_text = category_index[id]['name']

            a = Object(class_text, 1.0, xmin, xmax, ymin, ymax)
            annotatedList.append(a)
        return annotatedList


    def detect(self, cvImage):
        classes, scores, boxes = self.detector.detect(cvImage)
        labels = []
        #TODO: labels?
        for i in range(0,len(classes)):
            labels.append(self.detector.get_label(classes[i]))
        detections = self.convertMsgToObject(labels, scores, boxes)
        return detections


    def convertMsgToObject(self, labels, scores, boxes):
        detections = []
        for i in range(0,len(labels)):
            d = Object(labels[i], scores[i],
                       boxes[i][1], boxes[i][3], boxes[i][0], boxes[i][2])
            detections.append(d)
        return detections


    def detectAndRecognize(self, cvImage):
        height, width, channels = cvImage.shape
        #detect
        labels = []
        classes, scores, boxes = self.detector.detect(cvImage)
        for i in range(0,len(classes)):
            xmin = int(boxes[i][1] * width)
            xmax = int(boxes[i][3] * width)
            ymin = int(boxes[i][0] * height)
            ymax = int(boxes[i][2] * height)
            roi = cvImage[ymin:ymax, xmin:xmax]
            #TODO: directly from memory
            cv2.imwrite(filename=self._filename, img=roi)
            #recognize
            sorted_result = self.recognizer.recognize(self._filename)
            if sorted_result:
                best_label = sorted_result[-1][0]
                best_prob = sorted_result[-1][1]
            #TODO: unknown probability hardcoded
            best = CategoryProbability(label="unknown", probability=0.1)
            if best_prob > best.probability:
                best.label = best_label
                best.probability = best_prob
            # get label with highest probability
            #for p in r.categorical_distribution.probabilities:
            #    if p.probability > best.probability:
            #        best = p
            #TODO: handle labels differently?
            labels.append(best.label)
            scores[i] = best.probability
            if (best.label == "unkown"):
                print("unknown object found")
        detections = self.convertMsgToObject(labels, scores, boxes)
        return detections


    def evaluateImage(self, labelpath, imagepath, label_map, num_classes, doRecognition=True):
        cvImage = cv2.imread(imagepath, 3)
        annotatedList = self.readAnnotated(labelpath, label_map, num_classes)
        to_find = len(annotatedList)
        if doRecognition:
            detectedList = self.detectAndRecognize(cvImage)
        else:
            detectedList = self.detect(cvImage)
        correct, wrong, unkown_detected, image = evaluate.evaluateDetection(annotatedList, detectedList, self.threshold, cvImage)
        return correct, wrong, unkown_detected, to_find, image


    def evaluateGraphs(self, path, label_map, num_classes, doRecognition=True):
        index = 0
        for dirname, dirnames, filenames in os.walk(path):
            for filename in filenames:
                labelpath = dirname + '/' + filename
                if 'labels' in labelpath and '.txt' in labelpath:
                    imagepath = "{}/images/{}.jpg".format(dirname[:-7], filename[:-4])

                    if (os.path.isfile(imagepath)):
                        correct, wrong, unkown_detected, to_find, image = self.evaluateImage(labelpath, imagepath, label_map,
                                                                                        num_classes, doRecognition)
                        self.total_correct = self.total_correct + correct
                        self.total_wrong = self.total_wrong + wrong
                        self.total_unkown_detected = self.total_unkown_detected + unkown_detected
                        self.total_images = self.total_images + 1
                        self.total_to_find = self.total_to_find + to_find
                        save_filename = '{}/eval_image{}.jpg'.format(savepath, index)
                        # print("save image to: ", save_filename)
                        cv2.imwrite(save_filename, image)
                        index = index + 1


if __name__ == "__main__":

    # check for correct argument size
    if not len(sys.argv) >= 4:
        print '\033[91m' + 'Argument Error!\nUsage: python evaluateNet.py path_to_evalset path_to_graph path_to_labelmap path_to_rec_db num_classes [save_image_path]' + '\033[0m'
        exit(1)

    # check directory arguments
    if not os.path.isdir(sys.argv[1]):
        print '\033[91m' + sys.argv[1] + ' is not a directory!' + '\033[0m'
        exit(1)
    if not os.path.isdir(sys.argv[4]):
        print '\033[91m' + sys.argv[4] + ' is not a directory!' + '\033[0m'
        exit(1)


    # parse arguments
    path = sys.argv[1]
    label_map = label_map_util.load_labelmap(path+"/labelMap.pbtxt")
    graph_d=sys.argv[2]
    label_map_d=sys.argv[3]
    graph_r=sys.argv[4]+"/output_graph.pb"
    labels_r=sys.argv[4]+"/output_labels.txt"
    num_classes = int(sys.argv[5])

    if (len(sys.argv) == 7):
        savepath = sys.argv[6]
    else:
        savepath = "/tmp"

    if (len(sys.argv) == 8):
        doRecognition = sys.argv[7]
    else:
        doRecognition = True


    threshold = 0.3
    print("save eval images in {}".format(savepath))
    print("treshold: {}".format(threshold))
    print("doRecognition = {}".format(doRecognition))

    eval = EvaluateNet(threshold)



#    for detectiongraphs
 #       loaddetectiongraph
  #      if doRecognition:
   #         for recognitiongraphs
    #            loadrecognitiongraph
     #           evaluateImage
      #  else:


    eval.recognizer.load_graph(graph_r,labels_r)
    eval.detector.load_graph(graph_d,label_map_d)
    eval.evaluateGraphs(path, label_map, num_classes, True)

    print("detected ", eval.total_correct, " correct and ", eval.total_wrong, " wrong of ", eval.total_to_find, " objects,")
    print("detected ", eval.total_unkown_detected, " unknown objects in ", eval.total_images, " images")
    print '\033[1m\033[92mDone!\033[0m'
