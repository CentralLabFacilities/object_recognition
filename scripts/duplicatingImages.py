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
import random
from random import randint

from image_recognition_util.objectset_utils import ObjectsetUtils
from image_recognition_util.objectset_utils import BoundingBox

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
    train_txt.write(path + '\n')
    utils.writeAnnotationFile(path.replace("/images/", "/labels/").replace(".jpg", ".txt"), labels, boxes, image)

def show(img,time):
	cv2.imshow('img', img)
	cv2.waitKey(time)

def getRandomPositionOnSurface(w, h, bgAnnotationList):
    # randomly choose one surface annotation
    l = len(bgAnnotationList)
    r = randint(0, l - 1)
    surfaceBox = bgAnnotationList[r]

    limitLeft = surfaceBox.xmin + w
    limitRight = surfaceBox.xmax
    limitDown = surfaceBox.ymax
    limitUp = max(h, surfaceBox.ymin)

    if (limitRight < limitLeft or limitUp > limitDown):
        print("Surface too small to place roi.")
        return None

    bboxRand = BoundingBox(0, 0, 0, 0)

    # choose random point in surface range (bottom of object)
    bboxRand.ymax = randint(limitUp, limitDown)
    bboxRand.xmax = randint(limitLeft, limitRight)

    # set other coordinates according to roi size
    bboxRand.xmin = bboxRand.xmax - w
    bboxRand.ymin = bboxRand.ymax - h

    return bboxRand

def placeRoiOnBackground(fgCut, maskCut, bg, bbox, new_path, bg_index):

    # cut out object and background based on mask inside the roi
    maskInv = cv2.bitwise_not(maskCut)
    fg = cv2.bitwise_and(fgCut, fgCut, mask=maskCut)
    try:
        bgCut = bg[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax]
        bgCut = cv2.bitwise_and(bgCut, bgCut, mask=maskInv)
        roi = bgCut + fg
        # insert roi in large image
        h, w, c = bg.shape
        new = np.zeros((h, w, c), np.uint8)
        new[0:h, 0:w] = bg
        new[bbox.ymin:bbox.ymin + roi.shape[0], bbox.xmin:bbox.xmin + roi.shape[1]] = roi
        tmp_path = new_path[0:(len(new_path) - 4)] + "_bg" + str(bg_index).replace('.', '') + new_path[(len(new_path) - 4):]
        return new, tmp_path
    except:
        print("exception with mask-shape: {} and bg-shape: {}".format(maskCut.shape, bgCut.shape))


# change the brightness of the image with the factors in the lightning array
def change_lighting(image, new_path, min, max):
    max_image = np.zeros(image.shape) + 255   # makes sure the values won't be higher then 255
    lf = random.uniform(min, max)
    tmp_image = image.copy() * lf
    tmp_image = np.fmin(tmp_image, max_image)
    tmp_path = new_path[0:(len(new_path)-4)] + "_l" + str(lf).replace('.', '') + new_path[(len(new_path)-4):]
    return tmp_image, tmp_path


# simply mirror the image and the bounding boxes
def mirror_image(image, bbox, new_path):
    tmp_image = cv2.flip(image,1)
    tmp_bbox = copy.deepcopy(bbox)
    tmp_bbox.xmax = abs(bbox.xmax-1)
    tmp_bbox.xmin = abs(bbox.xmin-1)
    tmp_path = new_path[0:(len(new_path)-4)] + "_m" + new_path[(len(new_path)-4):]
    return tmp_image, tmp_path, tmp_bbox


# shrink the image to make the object smaller
def change_scale(image, bbox, new_path, min, max):
    sf = random.uniform(min, max)
    tmp_image = image.copy()
    tmp_bbox = copy.deepcopy(bbox)

    h, w, _ = tmp_image.shape
    tmp_image = misc.imresize(tmp_image, sf)
    new_h, new_w, _ = tmp_image.shape
    offset_h = int((h-new_h) / 2)
    offset_w = int((w-new_w) / 2)
    tmp_image = np.pad(tmp_image, ((offset_h, offset_h+(new_h%2)), (offset_w, offset_w+(new_w%2)), (0, 0)),
                       'constant', constant_values=255)
    tmp_bbox.xmax = (offset_w + (bbox.xmax * new_w)) / float(w)
    tmp_bbox.xmin = (offset_w + (bbox.xmin * new_w)) / float(w)
    tmp_bbox.ymax = (offset_h + (bbox.ymax * new_h)) / float(h)
    tmp_bbox.ymin = (offset_h + (bbox.ymin * new_h)) / float(h)

    tmp_path = new_path[0:(len(new_path)-4)] + "_s" + str(sf).replace('.', '') + new_path[(len(new_path)-4):]
    return tmp_image, tmp_path, tmp_bbox


def rotate_image(image, mask, rotation, new_path):
    tmp_image = image.copy()
    tmp_mask = mask.copy()
    random_degree = float(randint(0, rotation - 1) % 360)

    tmp_image = ndimage.rotate(tmp_image, random_degree)
    tmp_mask = ndimage.rotate(tmp_mask, random_degree)

    bbox = utils.getBboxByMask(tmp_mask)

    fg_cut = tmp_image[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax]
    mask_cut = tmp_mask[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax]

    # set all values to 255 or 0
    for (x, y), value in np.ndenumerate(mask_cut):
        if mask_cut[x][y] > 200:
            mask_cut[x][y] = 255
        else:
            mask_cut[x][y] = 0

    h, w, _ = fg_cut.shape

    tmp_path = new_path[0:(len(new_path) - 4)] + "_r" + str(random_degree).replace('.0', '') + new_path[(len(new_path) - 4):]

    return fg_cut, mask_cut, h, w, tmp_path


# blurs the image
def blur_image(image, new_path, min, max):
    bf = randint(min, max)
    tmp_image = image.copy()
    tmp_image = cv2.blur(tmp_image, (bf, bf))
    tmp_path = new_path[0:(len(new_path)-4)] + "_b" + str(bf).replace('.', '') + new_path[(len(new_path)-4):]
    return tmp_image, tmp_path


if __name__ == "__main__":

    # check for correct argument size
    if not len(sys.argv) >= 8:
        print '\033[91m' + 'Argument Error!\nUsage: python duplicatingImages.py ' \
                           'path_to_imageset save_image_path bg_path <light> <scale> <blur> <rotate>' + '\033[0m'
        exit(1)

    if not os.path.isdir(sys.argv[1]):
        print '\033[91m' + sys.argv[1] + ' is not a directory!' + '\033[0m'
        exit(1)
    if not os.path.isdir(sys.argv[2]):
        print '\033[91m' + sys.argv[2] + ' is not a directory!' + '\033[0m'
        exit(1)
    if not os.path.isdir(sys.argv[3]):
        print '\033[91m' + sys.argv[3] + ' is not a directory!' + '\033[0m'
        exit(1)
    if '[' not in sys.argv[4] and ']' not in sys.argv[4]:
        print '\033[91m' + sys.argv[4] + ' is not a lighting array like [0.2,0.5,1.5]' + '\033[0m'
        exit(1)
    if '[' not in sys.argv[5] and ']' not in sys.argv[5]:
        print '\033[91m' + sys.argv[5] + ' is not a scaling array like [0.3,0.7]' + '\033[0m'
        exit(1)
    if '[' not in sys.argv[6] and ']' not in sys.argv[6]:
        print '\033[91m' + sys.argv[6] + ' is not a blurring array like [4,8,15]' + '\033[0m'
        exit(1)
    if '[' not in sys.argv[7] and ']' not in sys.argv[7]:
        print '\033[91m' + sys.argv[7] + ' is not a rotation array like [min,max]' + '\033[0m'
        exit(1)

    image_path = sys.argv[1]
    save_path = sys.argv[2]
    bg_path = sys.argv[3]

    bg_list = []
    for dirname, dirnames, filenames in os.walk(bg_path):
        for filename in filenames:
            file = dirname + '/' + filename
            if file.endswith(".jpg"):
                if not (os.path.isfile(file.replace("/images/", "/labels/").replace(".jpg", ".txt"))):
                    print("error: Label file does not exist! Skipping image.")
                    continue
                bg_list.append("{}".format(file))

    lighting_array = [float(x) for x in sys.argv[4][1:len(sys.argv[4]) - 1].split(',')]
    print "lightning array: " + str(lighting_array)

    scaling_array = [float(x) for x in sys.argv[5][1:len(sys.argv[5]) - 1].split(',')]
    print "scaling array: " + str(scaling_array)

    blurring_array = [int(x) for x in sys.argv[6][1:len(sys.argv[6]) - 1].split(',')]
    print "blurring array: " + str(blurring_array)

    rotation_array = [int(x) for x in sys.argv[7][1:len(sys.argv[7]) - 1].split(',')]
    print "rotation array: " + str(rotation_array)

    # rotation hardcoded
    rotation = 360

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

            label_list = False
            box_list = False
            if "mask" in file_path: # ignore masks
                continue

            label_path = file_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
            mask_path = file_path.replace(".jpg", "_mask.jpg")
            if not (os.path.isfile(label_path) and os.path.isfile(mask_path)): # ignore image if label or mask doesnt exist
                continue

            image = cv2.imread(file_path)
            mask = cv2.imread(mask_path, 0)
            label_list, box_list = read_labels(label_path) # necessary?
            new_path = file_path.replace(image_path, save_path)

            for i in range(0,2):
                fg_cut, mask_cut, h_cut, w_cut, rot_path = rotate_image(image, mask, rotation, new_path)

                for i in range(0,len(bg_list)):
                    bg_file = bg_list[i]
                    bg_label_path = bg_file.replace("/images/", "/labels/").replace(".jpg", ".txt")
                    bg = cv2.imread(bg_file, 1)
                    _, bg_box_list = read_labels(bg_label_path)
                    h, w, _ = bg.shape
                    for i in range(0, len(bg_box_list)):
                        bg_box_list[i] = utils.getAbsoluteRoiCoordinates(bg_box_list[i], w, h)
                    bbox_rand = getRandomPositionOnSurface(w_cut, h_cut, bg_box_list)
                    if not bbox_rand:
                        continue
                    bg_img, bg_path = placeRoiOnBackground(fg_cut, mask_cut, bg, bbox_rand, rot_path, i)
                    norm_box = utils.getNormalizedRoiCoordinates(bbox_rand, w, h)
                    for i in range(0,2):
                        l_img, l_path = change_lighting(bg_img, bg_path, 0.2, 1.7)
                        for i in range(0, 3):
                            s_img, s_path, s_box = change_scale(l_img, norm_box, l_path, 0.3, 1.0)
                            for i in range(0, 2):
                                b_img, b_path = blur_image(s_img, s_path, 1, 4)

                                b_box = s_box
                                if(randint(0,1) == 1):
                                    b_img, b_path, b_box = mirror_image(b_img, s_box, b_path)

                                #save
                                save_image(b_img, label_list, [b_box], b_path)
