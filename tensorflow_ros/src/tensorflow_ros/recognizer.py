# Tensorflow
import tensorflow as tf

import numpy as np

#sys
import operator


class Recognizer:

    def __init__(self):
        self.labels = []


    def load_graph(self, graph_path, label_path):
        with open(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
            with open(label_path, 'rb') as f:
                self.labels = f.read().split("\n")


    def recognize(self,filename):
        print "recognize"
        with tf.Session() as sess:
            """1. Get result tensor"""
            result_tensor = sess.graph.get_tensor_by_name("final_result:0")

            """2. Open Image and perform prediction"""
            predictions = []

            #TODO: read image from memory (use different tensor??)
            with open(filename, 'rb') as f:
                predictions = sess.run(result_tensor, {'DecodeJpeg/contents:0': f.read()})
                predictions = np.squeeze(predictions)

            """3. Construct list with labels and probabilities"""
            result = zip(list(range(len(predictions))), predictions)

        # return sorted list
        sorted_result = sorted(result, key=lambda x: x[1])
        return sorted_result