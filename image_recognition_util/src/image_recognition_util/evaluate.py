from object import Object
from object import BoundingBox
import cv2

#max distance of bounding box coordinated in pixel
maxDistX = 1.0
maxDistY = 1.0

def evaluateDetection(annotatedList, detectedList, threshold, image):
    num_correct = 0
    num_wrong = 0
    num_unknown_detected = 0
    height, width, channels = image.shape
    #draw annotated bboxes with color: black
    color = (0,0,0)
    for annotated in annotatedList:
        drawBbox(color, image, annotated)
    idList = []
    for i in range(0,len(detectedList)):
        detected = detectedList[i]
        if (detected.prob > threshold):
            for annotated in annotatedList:
                if (matchBoundingBoxes(detected, annotated)):
                    if (detected.label == annotated.label):
                        num_correct = num_correct + 1
                        #green
                        color = (0,255,0)
                        drawBbox(color, image, detected)
                    else:
                        num_wrong = num_wrong + 1
                        #red
                        color = (0, 0, 255)
                        drawBbox(color, image, detected)
                        #TODO unknown vs. known but false classified
                    #remove annotated from list to prevent double detections
                    idList.append(i)
    # draw boxes that didn't match any annotated object
    for i in range(0, len(detectedList)):
        detected = detectedList[i]
        if (i not in idList):
            #TODO if label = unknown -> color green -> num_correct++
            #blue
            color = (255,0,0)
            if (detected.label=="unknown"):
                color = (0,255,0)
            drawBbox(color, image, detected)
            num_unknown_detected = num_unknown_detected + 1
    return num_correct, num_wrong, num_unknown_detected, image

def drawBbox(color, image, detected):
    height, width, channels = image.shape
    cv2.rectangle(image, (int(detected.bbox.xmin * width), int(detected.bbox.ymin * height)),
                  (int(detected.bbox.xmax * width), int(detected.bbox.ymax * height)), color, 2)
    image_label = '%s %f' % (detected.label, detected.prob)
    cv2.putText(image, image_label, (int(detected.bbox.xmin * width), int(detected.bbox.ymin * height)), 0, 0.6, color,
                1)

def matchBoundingBoxes(detected, annotated):

    #Detected vars
    xmax_det = detected.bbox.xmax
    xmin_det = detected.bbox.xmin
    ymax_det = detected.bbox.ymax
    ymin_det = detected.bbox.ymin
    width_det = xmax_det - xmin_det
    height_det = ymax_det - ymin_det
    area_det = width_det * height_det



    #Annotated vars
    xmax_an = annotated.bbox.xmax
    xmin_an = annotated.bbox.xmin
    ymax_an = annotated.bbox.ymax
    ymin_an = annotated.bbox.ymin
    width_an = xmax_an - xmin_an
    height_an = ymax_an - ymin_an
    area_an = width_an * height_an

    innerArea = max(0, min(xmax_det, xmax_an) - max(xmin_det, xmin_an)) * max(0, min(ymax_det, ymax_an) - max(ymin_det, ymin_an))
    outerArea = area_an + area_det - (2 * innerArea)
    if (innerArea == 0):
        ratio = 5
    else:
        ratio = outerArea / float(innerArea)  # the smaller the better

    # relative distances
    maxDistX = area_an*0.5
    maxDistY = area_an*0.5
    # alternatively match centroids of bboxes
    # or evaluate overlapping area of bboxes

    if (ratio < 2):
        return True


    if (inRange(xmin_det,xmin_an,maxDistX) and inRange(ymin_det,ymin_an,maxDistY)
        and inRange(xmax_det,xmax_an,maxDistX) and inRange(ymax_det,ymax_an,maxDistY)):
        return True
    else:
        return False


def inRange(a,b,maxDist):
    dist = abs(a - b)
    if (dist <= maxDist):
        return True
    else:
        return False

