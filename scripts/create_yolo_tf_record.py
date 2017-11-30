"""
Convert annotation files from darknet format to tensorflow format
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
from PIL import Image

import tensorflow as tf

from object_detection.utils import dataset_util
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2


flags = tf.app.flags
flags.DEFINE_string('input_path', '', 'Path to the annotation files and images')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def create_tf_example(labelpath,imagepath):
    with tf.gfile.GFile(os.path.join(imagepath), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = imagepath.encode('utf8')
    image_format = 'jpg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    #read label and normalized bbox shape from file (only one bbox per image here)
    with open(labelpath) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        content = content[0].split(' ')

    classes.append(int(content[0]))
    classes_text.append(content[0])
    xmins.append(float(content[1]))
    xmaxs.append(float(content[2]))
    ymins.append(float(content[3]))
    ymaxs.append(float(content[4]))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    path = os.path.join(FLAGS.input_path)

    # create label_map
    label_map_output = "{}/labelMap.pbtxt".format(FLAGS.output_path)
    class_name_path = "{}/classNames.txt".format(path)
    with open(class_name_path) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        label_map_string = ""
        for i in range(0, len(content)):
            label = content[i]
            label_map_string = label_map_string+"\nitem {\n\tid:"+str(i)+"\n\tname:'"+label+"'\n}"
    with tf.gfile.Open(label_map_output, 'wb') as f:
        f.write(label_map_string)
    print('Successfully created the label map: {}'.format(label_map_output))

    # create tf record
    tf_record_output = "{}/tf_train.record".format(FLAGS.output_path)
    writer = tf.python_io.TFRecordWriter(tf_record_output)

    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            labelpath = dirname + '/' + filename
            if 'labels' in labelpath and '.txt' in labelpath:
                imagepath = "{}/images/{}.jpg".format(dirname[:-7],filename[:-4])
                if (os.path.isfile(imagepath)):
                    print(imagepath)
                    tf_example = create_tf_example(labelpath,imagepath)
                    writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), tf_record_output)
    print('Successfully created the TFRecords: {}'.format(output_path))



if __name__ == '__main__':
  tf.app.run()
