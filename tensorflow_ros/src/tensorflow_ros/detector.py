import numpy as np

# Tensorflow
import tensorflow as tf
from object_detection.utils import label_map_util


class Detector:
    def __init__(self, num_classes=99, detection_threshold=0.5):
        self.detection_threshold = detection_threshold
        self.label_map = None
        self.categories = None
        self.category_index = None
        self.num_classes = num_classes

    def load_graph(self, pathToCkpt, pathToLabels):
        print pathToCkpt
        # load a (frozen) tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(pathToCkpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        print "load label map"
        print pathToLabels
        # loading label map
        self.label_map = label_map_util.load_labelmap(pathToLabels)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.num_classes,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def detect(self, image_np):
        print "detect"
        if self.detection_graph == None:
            print "No graph defined. You need to load a graph before detecting objects!"
            return None
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        with self.detection_graph.as_default():
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (self.boxes, self.scores, self.classes, self.num) = sess.run([
                    detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
        boxes = []
        scores = []
        classes = []
        # filter results with low scores
        for i in range(0,len(self.scores[0])):
            if self.scores[0][i] >= self.detection_threshold:
                scores.append(self.scores[0][i])
                boxes.append(self.boxes[0][i])
                classes.append(self.classes[0][i])
                print "found ", self.get_label(self.classes[0][i]), " with score ", self.scores[0][i], "at: ", self.boxes[0][i]

        return classes, scores, boxes

    def get_label(self, classId):
        return self.category_index[classId]['name']




