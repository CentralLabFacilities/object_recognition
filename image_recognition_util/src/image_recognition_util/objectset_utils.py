import os
import sys
from image_recognition_util.object import Object
from image_recognition_util.object import BoundingBox

class ObjectsetUtils():

    def __init__(self):
        print "init"

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