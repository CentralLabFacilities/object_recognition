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
        cv2.rectangle(image, (int(annotated.bbox.xmin * width), int(annotated.bbox.ymin * height)),
                      (int(annotated.bbox.xmax * width), int(annotated.bbox.ymax * height)), color, 2)
        #cv2.putText(image, annotated.label, (int(annotated.bbox.xmin* width), int(annotated.bbox.ymin * height)), 0, 0.4, (0, 0, 255), 1)

    for detected in detectedList:
        if (detected.prob > threshold):
            for annotated in annotatedList:
                if (matchBoundingBoxes(detected, annotated)):
                    if (detected.label == annotated.label):
                        num_correct = num_correct + 1
                        #green
                        color = (0,255,0)
                        drawBbox(color, image, detected)
                        print("*** correct detection ***")
                    else:
                        num_wrong = num_wrong + 1
                        #red
                        color = (0, 0, 255)
                        drawBbox(color, image, detected)
                        print(" wrong label ")
                    #remove annotated from list to prevent double detections
                    detectedList.remove(detected)
                    break
    # draw boxes that didn't match any annotated object
    for detected in detectedList:
        #if label = unknown -> color green -> num_correct++
        #blue
        color = (255,0,0)
        drawBbox(color, image, detected)
        num_unknown_detected = num_unknown_detected + 1
    return num_correct, num_wrong, num_unknown_detected, image

def drawBbox(color, image, detected):
    height, width, channels = image.shape
    cv2.rectangle(image, (int(detected.bbox.xmin * width), int(detected.bbox.ymin * height)),
                  (int(detected.bbox.xmax * width), int(detected.bbox.ymax * height)), color, 2)
    image_label = '%s %f' % (detected.label, detected.prob)
    cv2.putText(image, image_label, (int(detected.bbox.xmin * width), int(detected.bbox.ymin * height)), 0, 0.4, color,
                1)

def matchBoundingBoxes(detected, annotated):
    aWidth = annotated.bbox.xmax - annotated.bbox.xmin
    aHeight = annotated.bbox.ymax - annotated.bbox.ymin
    maxDistX = aWidth*0.3
    maxDistY = aHeight*0.3

    if (inRange(detected.bbox.xmin,annotated.bbox.xmin,maxDistX) and inRange(detected.bbox.ymin,annotated.bbox.ymin,maxDistY)
        and inRange(detected.bbox.xmax,annotated.bbox.xmax,maxDistX)
        and inRange(detected.bbox.ymax,annotated.bbox.ymax,maxDistY)):
        #bboxes match
        return True
    else:
        return False


def inRange(a,b,maxDist):
    dist = abs(a - b)
    if (dist <= maxDist):
        return True
    else:
        return False

