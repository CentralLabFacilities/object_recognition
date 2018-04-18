# openCV
import cv2

# sys
import sys
import os

# tensorflow
import tensorflow as tf

# object rec
from image_recognition_util import evaluate
from image_recognition_util.object import Object
from image_recognition_util.objectset_utils import ObjectsetUtils
from tensorflow_ros import detector
from tensorflow_ros import recognizer



flags = tf.app.flags
flags.DEFINE_string('testset', '', 'Path to testset.')
flags.DEFINE_string('detect_graph', '', 'Path to detection graph.')
flags.DEFINE_string('labels', '', 'Path to labelMap.')
flags.DEFINE_string('rec_graph', '', 'Path to recognition graph.')
flags.DEFINE_string('logdir', '/tmp', 'Save location for logs and images.')
flags.DEFINE_integer('num_classes', 99, 'number of classes')
flags.DEFINE_float('threshold', 0.5, 'detection threshold')
flags.DEFINE_boolean('save_images', False, 'Save evaluation images.')
flags.DEFINE_boolean('recognize', True, 'Do recognition for each detection.')

FLAGS = flags.FLAGS


class EvaluateNet:
    def __init__(self, threshold):
        # variables
        self.threshold = threshold
        self.total_correct = 0
        self.total_wrong = 0
        self.total_images = 0
        self.total_unkown_detected = 0
        self.total_to_find = 0
        self.index = 0
        self.log_str = ""

        # objects
        print("initialize detector with threshold {}".format(threshold))
        self.detector = detector.Detector(detection_threshold=self.threshold)
        self.recognizer = recognizer.Recognizer()
        self.util = ObjectsetUtils()



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
            if (abs(ymin-ymax) <= 0 or abs(xmin-xmax) <= 0):
                print "ERROR: roi size is 0"
                return None
            # TODO: directly from memory
            # imgpath = "{}/img{}.jpg".format(self._filepath,datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_%f"))
            # print imgpath
            imgpath = "/tmp/rec_img.jpg"
            cv2.imwrite(filename=imgpath, img=roi)
            # recognize
            sorted_result = self.recognizer.recognize(imgpath)
            if sorted_result:
                tmp_label = sorted_result[-1][0]
                tmp_prob = sorted_result[-1][1]
            # TODO: unknown probability hardcoded
            best_label = "unknown"
            best_prob = 0.1
            if tmp_prob > best_prob:
                best_label = tmp_label
                best_prob = tmp_prob
            # print old and new label
            print("d: {} -> r: {}".format(self.detector.get_label(classes[i]), best_label))
            # get label with highest probability
            # for p in r.categorical_distribution.probabilities:
            #    if p.probability > best_probability:
            #        best = p
            # TODO: handle labels differently?
            labels.append(best_label)
            scores[i] = best_prob
            if (best_label == "unkown"):
                print("unknown object found")
        detections = self.convertMsgToObject(labels, scores, boxes)
        return detections

    def numObjectsToFind(self, annotatedList):
        to_find = 0
        for annotated in annotatedList:
            if not (annotated.label == "unknown"):
                to_find = to_find + 1
        return to_find

    def evaluateImage(self, labelpath, imagepath, label_map_path, num_classes, savepath, doRecognition=True):
        cvImage = cv2.imread(imagepath, 3)
        annotatedList = self.util.readAnnotated(labelpath, label_map_path, num_classes)
        to_find = self.numObjectsToFind(annotatedList)
        if doRecognition:
            detectedList = self.detectAndRecognize(cvImage)

        else:
            detectedList = self.detect(cvImage)

        if (detectedList == None):
            return 0,0,0,0,cvImage

        correct, wrong, unkown_detected, image = evaluate.evaluateDetection(annotatedList, detectedList, self.threshold, cvImage, savepath)
        return correct, wrong, unkown_detected, to_find, image


    def evaluateGraphs(self, path, label_map_path, num_classes, savepath, doRecognition=True, saveImages=True):
        index = 0
        for dirname, dirnames, filenames in os.walk(path):
            for filename in filenames:
                labelpath = dirname + '/' + filename
                if 'labels' in labelpath and '.txt' in labelpath:
                    imagepath = "{}/images/{}.jpg".format(dirname[:-7], filename[:-4])

                    if (os.path.isfile(imagepath)):
                        correct, wrong, unkown_detected, to_find, image = self.evaluateImage(labelpath, imagepath, label_map_path,
                                                                                        num_classes, savepath, doRecognition)
                        self.total_correct = self.total_correct + correct
                        self.total_wrong = self.total_wrong + wrong
                        self.total_unkown_detected = self.total_unkown_detected + unkown_detected
                        self.total_images = self.total_images + 1
                        self.total_to_find = self.total_to_find + to_find

                        if saveImages:
                            save_filename = '{}/eval_image{}.jpg'.format(logdir, index)
                            cv2.imwrite(save_filename, image)
                            index = index + 1

    def printAndLog(self, graph_path, logdir):
        filename = logdir+"/log.txt"

        total_detected = eval.total_correct+eval.total_wrong
        print("total detected: {}".format(total_detected))
        recognize_percent = 0.0
        detect_percent = 0.0
        if not (total_detected == 0):
            recognize_percent = float(eval.total_correct)
            recognize_percent = recognize_percent/float(total_detected)
        if not (eval.total_to_find == 0):
            detect_percent = float(total_detected)
            detect_percent = detect_percent/float(eval.total_to_find)

        print_str = graph_path + "\nevaluated with {} images".format(eval.total_images) \
                  + "\ndetections: {} of {} ({})".format(total_detected,eval.total_to_find, detect_percent) \
                  + "\ncorrect labels: {} of {} ({})".format(eval.total_correct, total_detected, recognize_percent) \
                  + "\nunknown detections: {} ({} per image)\n".format(eval.total_unkown_detected, (eval.total_unkown_detected/eval.total_images))
        print(print_str)
        unknown_percent = float(eval.total_unkown_detected) / float(eval.total_images)
        self.log_str = self.log_str + str(detect_percent) + "\t" + str(recognize_percent) + "\t" + str(unknown_percent) + "\t" + graph_path + "\n"


        print("save log at: {}".format(filename))
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        label_file = open(filename, 'w')
        label_file.write(self.log_str)
        #reset variables
        self.total_correct = 0
        self.total_wrong = 0
        self.total_images = 0
        self.total_unkown_detected = 0
        self.total_to_find = 0


if __name__ == "__main__":

    # check if flags are set
    if not (FLAGS.testset and FLAGS.detect_graph and FLAGS.labels and FLAGS.rec_graph):
        print '\033[91m' + 'Usage: python evaluateNet.py [args]\n' \
                           'necessary:\n' \
                           '\t--testset\n' \
                           '\t--detect_graph\n' \
                           '\t--labels\n' \
                           '\t--rec_graph\n' \
                           'optional:\n' \
                           '\t--num_classes\t default: 99\n' \
                           '\t--threshold\t default: 0.5\n' \
                           '\t--save_images\t default: False\n' \
                           '\t--recognize\t default: True\n' \
                           '\t--logdir\t default: /tmp\n' \
                           '\033[0m'
        exit(1)

    # get flag values
    testset = FLAGS.testset
    graph_d = FLAGS.detect_graph
    graph_r = FLAGS.rec_graph + "/output_graph.pb"
    labels_r = FLAGS.rec_graph + "/output_labels.txt"
    label_map_d = FLAGS.labels

    # check directories and files
    if not os.path.isdir(testset):
        print '\033[91m' + testset + ' is not a directory!' + '\033[0m'
        exit(1)
    if not os.path.isdir(graph_d):
        print '\033[91m' + graph_d + ' is not a directory!' + '\033[0m'
        exit(1)
    if not os.path.isfile(label_map_d):
        print '\033[91m File ' + label_map_d + ' does not exist!' + '\033[0m'
        exit(1)
    if not os.path.isfile(graph_r):
        print '\033[91m File ' + graph_r + ' does not exist!' + '\033[0m'
        exit(1)
    if not os.path.isfile(labels_r):
        print '\033[91m File ' + labels_r + ' does not exist!' + '\033[0m'
        exit(1)
    if not os.path.isfile(testset+"/labelMap.pbtxt"):
        print '\033[91m File ' + testset+ '/labelMap.pbtxt does not exist!' + '\033[0m'
        exit(1)

    # get optional flags
    num_classes = FLAGS.num_classes
    save_images = FLAGS.save_images
    logdir = FLAGS.logdir
    doRecognition = FLAGS.recognize
    threshold = FLAGS.threshold

    # load label map for testset
    label_map_path = testset+"/labelMap.pbtxt"

    # print settings
    print("save eval images and logs in {}".format(logdir))
    print("save images = {}".format(save_images))
    print("treshold: {}".format(threshold))
    print("doRecognition = {}".format(doRecognition))
    print("num classes = {}".format(num_classes))

    # eval object
    eval = EvaluateNet(threshold)

    # load recognition graph
    eval.recognizer.load_graph(graph_r,labels_r)
    save_rec_image_path = logdir + '/rec_img'
    if not os.path.exists(save_rec_image_path):
        os.makedirs(save_rec_image_path)

    # iterate through all detections graphs in the given directory, then load and evaluate
    for dirname, dirnames, filenames in os.walk(graph_d):
        for filename in filenames:
            filepath = dirname + '/' + filename
            if 'frozen_inference_graph' in filename:

                print ("evaluate graph: {}".format(filepath))
                eval.detector.load_graph(filepath, label_map_d)
                eval.evaluateGraphs(testset, label_map_path, num_classes, save_rec_image_path, doRecognition, save_images)
                eval.printAndLog(filepath, logdir)

    #eval.detector.load_graph(graph_d,label_map_d)
    #eval.evaluateGraphs(testset, label_map, num_classes, True)

    print '\033[1m\033[92mDone!\033[0m'
