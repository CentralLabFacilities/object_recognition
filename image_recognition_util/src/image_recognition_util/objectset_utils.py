import os
import sys
from image_recognition_util.object import Object
from image_recognition_util.object import BoundingBox
from object_detection.utils import label_map_util

class ObjectsetUtils():

    def __init__(self):
        print "init"

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

    def getAbsoluteRoiCoordinates(self, bbox, w, h):
        absBbox = BoundingBox(0,0,0,0)
        absBbox.xmin = int(bbox.xmin * w)
        absBbox.xmax = int(bbox.xmax * w)
        absBbox.ymin = int(bbox.ymin * h)
        absBbox.ymax = int(bbox.ymax * h)
        return absBbox

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