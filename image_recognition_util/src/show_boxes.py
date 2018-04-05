import cv2
import os
import sys

from image_recognition_util.src.image_recognition_util.objectset_utils import ObjectsetUtils


def show_bbox(imagepath, labelpath, util):
    image = cv2.imread(imagepath,3)
    height, width, channels = image.shape

    x_min, y_min, x_max, y_max = util.getNormalizedRoiFromYolo(labelpath)

    cv2.rectangle(image, (int(x_min*width), int(y_min*height)),(int(x_max*width), int(y_max*height)), (0, 100, 200), 3)
    cv2.imshow('image', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit(0)

if __name__ == "__main__":

    # check for correct argument size
    if not len(sys.argv) == 2:
        print '\033[91m' + 'Argument Error!\nUsage: python fix_txt.py path_to_dataset' + '\033[0m'
        exit(1)

    # check if argument given is a directory
    if not os.path.isdir(sys.argv[1]):
        print '\033[91m' + sys.argv[1] + ' is not a directory!' + '\033[0m'
        exit(1)

    path = sys.argv[1]

    util = ObjectsetUtils()

    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            labelpath = dirname + '/' + filename
            if 'labels' in labelpath and '.txt' in labelpath:
                imagepath = "{}/images/{}.jpg".format(dirname[:-7],filename[:-4])
                if (os.path.isfile(imagepath)):
                    show_bbox(imagepath, labelpath, util)

    cv2.destroyAllWindows()
    print '\033[1m\033[92mDone!\033[0m'
