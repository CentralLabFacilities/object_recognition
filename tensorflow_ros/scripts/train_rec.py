# tensorflow
import tensorflow as tf


from tensorflow_ros import retrain, utils
#import webbrowser
#import subprocess
#import time
#import signal

flags = tf.app.flags
flags.DEFINE_string('input', '', 'input dir')
flags.DEFINE_string('output', '/tmp', 'output dir')
flags.DEFINE_integer('steps', 1000, 'training steps')
flags.DEFINE_integer('batch', 10, 'batch size')



FLAGS = flags.FLAGS


def trainInception(input_dir, output_dir, steps, batch):
    print("output_dir = "+output_dir)
    model_dir = output_dir+"/inception"

    utils.maybe_download_and_extract("http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz",
                                     model_dir)

    try:
        retrain.main(input_dir, model_dir, output_dir,
                     steps=steps, batch=batch)
        print("Retrain succes", "Succesfully retrained the top layers! Check Tensorboard for the results!")
        print("Training done :)")
    except Exception as e:
        print("Retrain failed. Something went wrong during retraining, '%s'" % str(e))

if __name__ == '__main__':
    # check if flags are set
    if not (FLAGS.input):
        print '\033[91m' + 'Usage: python evaluateNet.py [args]\n' \
                           'necessary:\n' \
                           '\t--input\n' \
                           'optional:\n' \
                           '\t--output\t default: /tmp\n' \
                           '\t--steps\t default: 1000\n' \
                           '\t--batch\t default: 10\n' \
                           '\t--flip_left_right\t default: False\n' \
                           '\t--random_crop\t default: 0\n' \
                           '\t--random_scale\t default: 0\n' \
                           '\t--random_brightness\t default: 0\n' \
                           '\033[0m'
        exit(1)

    # get flag values
    input_dir = FLAGS.input
    output_dir = FLAGS.output
    steps = FLAGS.steps
    batch = FLAGS.batch


    trainInception(input_dir, output_dir, steps, batch)