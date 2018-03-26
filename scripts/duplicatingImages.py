# openCV
import cv2
# sys
import sys
import os
import copy
import imghdr
# math
import numpy as np
import scipy.misc as misc
from scipy import ndimage
from random import randint

from image_recognition_util.objectset_utils import ObjectsetUtils

# global variable, sorry for that
train_txt = False
utils = ObjectsetUtils()


# opens label files and returns the data
def read_labels(labels_path):
    labelList = utils.getLabelList(labels_path)
    bboxList = utils.getRoiList(labels_path)

    return labelList, bboxList


# delete soon
def save_labels(labels_path, label, boxes):
    new_file = open(labels_path, 'w+')
    content = ""
    for i in range(len(label)):
        content += str(label[i])
        for x in boxes[i]:
            content = content + " " + str(x)
        content += '\n'

    new_file.write(content)
    new_file.close()


# save image and label file to the new location
def save_image(image, labels, boxes, path):
    cv2.imwrite(path, image)
    if train_txt:
        train_txt.write(path + '\n')
    if label:
        utils.writeAnnotationFile(path.replace("/images/", "/labels/").replace(".jpg", ".txt"), labels, boxes, image)


# change the brightness of the image with the factors in the lightning array
def change_lighting(image, labels, boxes, new_path, lighting_array):
    for i in lighting_array:
        max_image = np.zeros(image.shape) + 255   # makes sure the values won't be higher then 255
        tmp_image = image.copy() * i
        tmp_image = np.fmin(tmp_image, max_image)
        tmp_path = new_path[0:(len(new_path)-4)] + "_l" + str(i).replace('.', '') + new_path[(len(new_path)-4):]
        save_image(tmp_image, labels, boxes, tmp_path)


# simply mirror the image and the bounding boxes
def mirror_image(image, labels, boxes, new_path):
    tmp_image = np.fliplr(image)
    tmp_boxes = copy.deepcopy(boxes)
    for bbox in tmp_boxes:
        bbox.xmax = abs(bbox.xmax-1)
        bbox.xmin = abs(bbox.xmin-1)
    tmp_path = new_path[0:(len(new_path)-4)] + "_m" + new_path[(len(new_path)-4):]
    save_image(tmp_image, labels, tmp_boxes, tmp_path)


# shrink the image to make the object smaller
def change_scale(image, labels, boxes, new_path, scaling_array):
    for i in scaling_array:
        tmp_image = image.copy()
        tmp_boxes = copy.deepcopy(boxes)

        h, w, _ = tmp_image.shape
        tmp_image = misc.imresize(tmp_image, i)
        new_h, new_w, _ = tmp_image.shape
        offset_h = int((h-new_h) / 2)
        offset_w = int((w-new_w) / 2)
        tmp_image = np.pad(tmp_image, ((offset_h, offset_h+(new_h%2)), (offset_w, offset_w+(new_w%2)), (0, 0)),
                           'constant', constant_values=255)
        for bbox in tmp_boxes:
            bbox.xmax = (offset_w + (bbox.xmax * new_w)) / float(w)
            bbox.xmin = (offset_w + (bbox.xmin * new_w)) / float(w)
            bbox.ymax = (offset_h + (bbox.ymax * new_h)) / float(h)
            bbox.ymin = (offset_h + (bbox.ymin * new_h)) / float(h)

        tmp_path = new_path[0:(len(new_path)-4)] + "_s" + str(i).replace('.', '') + new_path[(len(new_path)-4):]
        save_image(tmp_image, labels, tmp_boxes, tmp_path)


def rotate_image(image, mask, rotation):
    tmp_image = image.copy()
    tmp_mask = mask.copy()
    random_degree = float(randint(0, rotation - 1) % 360)

    tmp_image = ndimage.rotate(tmp_image, random_degree)
    tmp_mask = ndimage.rotate(tmp_mask, random_degree)

    bbox = utils.getBboxByMask(tmp_mask)

    fgCut = tmp_image[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax]
    maskCut = tmp_mask[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax]

    h, w, _ = fgCut.shape()

    return fgCut, maskCut, h, w


# blurs the image
def blur_image(image, labels, boxes, new_path, blurring_array):
    for i in blurring_array:
        tmp_image = image.copy()
        tmp_image = cv2.blur(tmp_image, (i, i))
        tmp_path = new_path[0:(len(new_path)-4)] + "_b" + str(i).replace('.', '') + new_path[(len(new_path)-4):]
        save_image(tmp_image, labels, boxes, tmp_path)


if __name__ == "__main__":

    # check for correct argument size
    if not len(sys.argv) >= 6:
        print '\033[91m' + 'Argument Error!\nUsage: python duplicatingImages.py ' \
                           'path_to_imageset save_image_path [0.2,0.5,1.5] [0.3,0.7] [4,8,15]' + '\033[0m'
        exit(1)

    if not os.path.isdir(sys.argv[1]):
        print '\033[91m' + sys.argv[1] + ' is not a directory!' + '\033[0m'
        exit(1)
    if not os.path.isdir(sys.argv[2]):
        print '\033[91m' + sys.argv[2] + ' is not a directory!' + '\033[0m'
        exit(1)
    if '[' not in sys.argv[3] and ']' not in sys.argv[3]:
        print '\033[91m' + sys.argv[3] + ' is not a lighting array like [0.2,0.5,1.5]' + '\033[0m'
        exit(1)
    if '[' not in sys.argv[4] and ']' not in sys.argv[4]:
        print '\033[91m' + sys.argv[4] + ' is not a scaling array like [0.3,0.7]' + '\033[0m'
        exit(1)
    if '[' not in sys.argv[5] and ']' not in sys.argv[5]:
        print '\033[91m' + sys.argv[5] + ' is not a blurring array like [4,8,15]' + '\033[0m'
        exit(1)

    image_path = sys.argv[1]
    save_path = sys.argv[2]
    lighting_array = [float(x) for x in sys.argv[3][1:len(sys.argv[3]) - 1].split(',')]
    print "lightning array: " + str(lighting_array)

    scaling_array = [float(x) for x in sys.argv[4][1:len(sys.argv[4]) - 1].split(',')]
    print "scaling array: " + str(scaling_array)

    blurring_array = [int(x) for x in sys.argv[5][1:len(sys.argv[5]) - 1].split(',')]
    print "blurring array: " + str(blurring_array)

    for dirname, dirnames, filenames in os.walk(image_path):
        for filename in filenames:
            file_path = dirname + '/' + filename

            if not os.path.exists(dirname.replace(image_path, save_path)):  # creates dir path
                os.makedirs(dirname.replace(image_path, save_path))


            # deals with all None-Image files
            if (imghdr.what(file_path) == None):

                if ".jpg" in filename:   # delete empty images
                    continue

                if dirname.endswith("/labels"):  # ignores the label files, they will be written later
                    continue

                if "train.txt" in filename or "test.txt" in filename:   # leaves the files empty to fill it later
                    train_txt = open(file_path.replace(image_path, save_path).replace("test.txt", "train.txt"), 'a+')
                else:   # copy the files without changing
                    old_file = open(file_path, 'r')
                    new_file = open(file_path.replace(image_path, save_path), 'w+')
                    new_file.write(old_file.read())
                    old_file.close()  # close the streams
                    new_file.close()
                continue

            # image manipulation and saving
            label = False
            boxes = False
            image = cv2.imread(file_path)
            if "/images/" in file_path and "mask" not in filename:
                label, boxes = read_labels(file_path.replace("/images/", "/labels/").replace(".jpg", ".txt"))

            new_path = file_path.replace(image_path, save_path)

            if label:
                blur_image(image, label, boxes, new_path, blurring_array)

            # save image
            cv2.imwrite(new_path, image)

            if train_txt:
                train_txt.write(new_path + '\n')

            if label:
                save_labels(new_path.replace("/images/", "/labels/").replace(".jpg", ".txt"), label, boxes)

    if train_txt:
        train_txt.close()
