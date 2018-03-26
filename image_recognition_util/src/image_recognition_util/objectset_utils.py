import os
import sys
import cv2
from image_recognition_util.object import Object
from image_recognition_util.object import BoundingBox
from object_detection.utils import label_map_util

class ObjectsetUtils():

    def __init__(self):
        print "init object set utils"

    def convert(self, size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box.xmin + box.xmax) / 2.0
        y = (box.ymin + box.ymax) / 2.0
        w = box.xmax- box.xmin
        h = box.ymax - box.ymin
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    def readAnnotated(self, labelpath, label_map, num_classes):
        annotatedList = []
        # "label" is the id
        labelList = self.getLabelList(labelpath)
        bboxList = self.getRoiList(labelpath)

        # get label
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                    use_display_name=True)
        for i in range(0, len(labelList)):
            label = labelList[i]
            xmin = bboxList[i].xmin
            xmax = bboxList[i].xmax
            ymin = bboxList[i].ymin
            ymax = bboxList[i].ymax
            category_index = label_map_util.create_category_index(categories)
            id = int(label) + 1
            class_text = category_index[id]['name']

            a = Object(class_text, 1.0, xmin, xmax, ymin, ymax)
            annotatedList.append(a)
        return annotatedList

    def writeAnnotationFile(self, labelpath, idList, boxList, image):
        if not (len(idList) == len(boxList)):
            print("error: list size mismatch (idList: {}, boxList: {}".format(len(idList),len(boxList)))
            return
        label_str = ""
        for i in range(0,len(idList)):
            bbox = boxList[i]
            id = idList[i]

            # convert bbox for darknet format
            h, w = image.shape[:2]
            bb = self.convert((w, h), bbox)

            # write converted bbox as label in label_dir
            if id is not None:
                label_str = label_str + (str(id) + " " + " ".join([str(a) for a in bb]) + '\n')

        label_file = open(labelpath, 'w+')
        label_file.write(label_str)

    def getAbsoluteRoiCoordinates(self, normBox, w, h):
        absBbox = BoundingBox(0,0,0,0)
        absBbox.xmin = int(normBox.xmin * w)
        absBbox.xmax = int(normBox.xmax * w)
        absBbox.ymin = int(normBox.ymin * h)
        absBbox.ymax = int(normBox.ymax * h)
        return absBbox

    def getNormalizedRoiCoordinates(self, absBox, w, h):
        normBox = BoundingBox(0,0,0,0)
        normBox.xmin = absBox.xmin/float(w)
        normBox.xmax = absBox.xmax/float(w)
        normBox.ymin = absBox.ymin/float(h)
        normBox.ymax = absBox.ymax/float(h)
        return normBox

    def getNormalizedRoiFromYolo(self,labelpath):
        with open(labelpath) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            content = content[0].split(' ')

            x_center = float(content[1])
            y_center = float(content[2])
            bbox_width = float(content[3])
            bbox_height = float(content[4])

            xmin = x_center - bbox_width / 2
            ymin = y_center - bbox_height / 2
            xmax = x_center + bbox_width / 2
            ymax = y_center + bbox_height / 2
        return xmin, ymin, xmax, ymax

    def getRoiList(self,labelpath):
        bboxList = []
        with open(labelpath) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            for i in range(0,len(content)):
                line = content[i].split(' ')
                x_center = float(line[1])
                y_center = float(line[2])
                bbox_width = float(line[3])
                bbox_height = float(line[4])

                xmin = x_center - bbox_width / 2
                ymin = y_center - bbox_height / 2
                xmax = x_center + bbox_width / 2
                ymax = y_center + bbox_height / 2

                bbox = BoundingBox(xmin, xmax, ymin, ymax)
                bboxList.append(bbox)

            return bboxList

    def getLabelIdFromYolo(self,labelpath):
        with open(labelpath) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            content = content[0].split(' ')

        return content[0]

    def getLabelList(self,labelpath):
        labelList = []
        with open(labelpath) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            for i in range(0, len(content)):
                line = content[i].split(' ')
                labelList.append(line[0])
        return labelList

    def getBboxByMask(self,mask):
        mask, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0
        largest_contour_index = 0
        for i in range(len(contours)):
            a = cv2.contourArea(contours[i], False)
            if a > largest_area:
                largest_area = a
                largest_contour_index = i
        box = cv2.boundingRect(contours[largest_contour_index])
        bbox = BoundingBox(box[0],box[0]+box[2],box[1],box[1]+box[3])
        return bbox