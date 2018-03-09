# openCV
import cv2
# sys
import sys
import os
import imghdr
# math
import numpy as np

# global variable, sorry for that
train_txt = False


# opens label files and returns the data
def read_labels(labels_path):
    label = []
    boxes = []

    with open(labels_path) as f:
        for line in f:
            list = line.split()
            label.append(int(list[0]))
            for x in range(4):
                boxes.append(float(list[x+1]))

    num_lines=len(boxes)/4
    boxes = np.array(boxes)
    boxes = boxes.reshape((num_lines,4))
    return label, boxes


# save changed labels to new file
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
        save_labels(path.replace("/images/", "/labels/").replace(".jpg", ".txt"), labels, boxes)


# change the brightness of the image with the factors in the lightning array
def change_lighting(image, labels, boxes, new_path, lighting_array):
    for i in lighting_array:
        max_image = np.zeros(image.shape) + 255   # makes sure the values won't be higher then 255
        tmp_image = image * i
        tmp_image = np.fmin(tmp_image, max_image)
        tmp_path = new_path[0:(len(new_path)-4)] + "_l" + str(i).replace('.', '') + new_path[(len(new_path)-4):]
        save_image(tmp_image, labels, boxes, tmp_path)


# simply mirror the image and the bounding boxes
def mirror_image(image, labels, boxes, new_path):
    tmp_image = np.fliplr(image)
    tmp_boxes = boxes.copy()
    for i in range(tmp_boxes.shape[0]):
        tmp_boxes[i][0] = abs(tmp_boxes[i][0]-1)
    tmp_path = new_path[0:(len(new_path)-4)] + "_m" + new_path[(len(new_path)-4):]
    save_image(tmp_image, labels, tmp_boxes, tmp_path)


if __name__ == "__main__":

    # check for correct argument size
    if not len(sys.argv) >= 4:
        print '\033[91m' + 'Argument Error!\nUsage: python duplicatingImages.py path_to_imageset save_image_path [0.2,0.4,1.5]' + '\033[0m'
        exit(1)

    if not os.path.isdir(sys.argv[1]):
        print '\033[91m' + sys.argv[1] + ' is not a directory!' + '\033[0m'
        exit(1)
    if not os.path.isdir(sys.argv[2]):
        print '\033[91m' + sys.argv[2] + ' is not a directory!' + '\033[0m'
        exit(1)

    if '[' not in sys.argv[3] and ']' not in sys.argv[3]:
        print '\033[91m' + sys.argv[3] + ' is not a lighting array like [0.2, 0.4, 1.5]' + '\033[0m'

    image_path = sys.argv[1]
    save_path = sys.argv[2]
    lighting_array = [float(x) for x in sys.argv[3][1:len(sys.argv[3]) - 1].split(',')]
    print "lightning array: " + str(lighting_array)

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
            if "/images/" in file_path:
                label, boxes = read_labels(file_path.replace("/images/", "/labels/").replace(".jpg", ".txt"))

            new_path = file_path.replace(image_path, save_path)

            if label:
                change_lighting(image, label, boxes, new_path, lighting_array)
                mirror_image(image, label, boxes, new_path)

            # save image
            cv2.imwrite(new_path, image)

            if train_txt:
                train_txt.write(new_path + '\n')

            if label:
                save_labels(new_path.replace("/images/", "/labels/").replace(".jpg", ".txt"), label, boxes)

    if train_txt:
        train_txt.close()
