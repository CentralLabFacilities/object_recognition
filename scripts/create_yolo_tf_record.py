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
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('input_path', '', 'Path to the annotation files and images')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def create_tf_example(labelpath,imagepath,label_map,num_classes):
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

    # read label-id and normalized bbox shape from file (only one bbox per image here)
    with open(labelpath) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        content = content[0].split(' ')

    # get class name from label map by id
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                     use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    id = int(content[0])+1
    class_text = category_index[id]['name']

    classes.append(id)
    classes_text.append(class_text.encode('utf8'))
    # in yolo format we have x_min, y_min, width, height
    xmins.append(float(content[1]))
    xmaxs.append(float(content[1])+float(content[2]))
    ymins.append(float(content[3]))
    ymaxs.append(float(content[3])+float(content[4]))

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
    num_classes = 0
    # create label_map
    print("expecting {}/classNames.txt".format(path))
    label_map_output = "{}/labelMap.pbtxt".format(FLAGS.output_path)
    class_name_path = "{}/classNames.txt".format(path)
    with open(class_name_path) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        label_map_string = ""
        num_classes = len(content)
        for i in range(0, num_classes):
            label = content[i]
            label_map_string = label_map_string+"\nitem {\n  id:"+str(i+1)+"\n  name:'"+label+"'\n}"
    with tf.gfile.Open(label_map_output, 'wb') as f:
        f.write(label_map_string)
    print('Successfully created the label map: {}'.format(label_map_output))

    # create tf record
    tf_record_output = "{}/tf_train.record".format(FLAGS.output_path)
    writer = tf.python_io.TFRecordWriter(tf_record_output)
    label_map = label_map_util.load_labelmap(label_map_output)

    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            labelpath = dirname + '/' + filename
            if 'labels' in labelpath and '.txt' in labelpath:
                imagepath = "{}/images/{}.jpg".format(dirname[:-7],filename[:-4])
                if (os.path.isfile(imagepath)):
                    #print(imagepath)
                    tf_example = create_tf_example(labelpath,imagepath, label_map, num_classes)
                    if (tf_example == None):
                        continue;
                    else:
                        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), tf_record_output)
    print('Successfully created the TFRecords: {}'.format(output_path))



if __name__ == '__main__':
  tf.app.run()
