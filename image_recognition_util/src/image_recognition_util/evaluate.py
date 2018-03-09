import cv2

#max distance of bounding box coordinated in pixel
maxDistX = 1.0
maxDistY = 1.0


def evaluateDetection(annotatedList, detectedList, threshold, image):
    num_correct = 0           # correct detected and recognized objects
    num_wrong = 0             # correct detected but wrong recognized objects
    num_wrong_detected = 0    # wrong detected objects

    # draw annotated bboxes with color: black
    color = (0, 0, 0)
    for annotated in annotatedList:
        drawBbox(color, image, annotated)


    # evaluation loop for each detected object
    for detected in detectedList:
        correct_recognized = False
        correct_detected = False  # false if none or to many annotations fit
        double_detected = False   # check if a better detection exist

        # check whether the object is correct detected and/or labeled
        for annotated in annotatedList:
            if (matchBoundingBoxes(detected, annotated) and detected.prob > threshold):
                correct_detected += 1
                if (detected.label == annotated.label):
                    correct_recognized = True

        # check if a better detection exist so it can be ignored
        for other_detected in detectedList:
            double_detected += doubleTest(detected, other_detected)

        # coloring the detection
        if (double_detected):
            color = (0, 255, 255)  # yellow
        elif (correct_detected):
            if (correct_recognized):
                num_correct += 1
                color = (0, 255, 0)    # green
            else:
                num_wrong += 1
                color = (0, 0, 255)    # red but why?
        elif (detected.label == "unknown"):
            num_correct += 1
            color = (0, 255, 0)  # green
        else:
            num_wrong_detected += 1
            color = (255, 0, 0)  # blue but why?

        drawBbox(color, image, detected)

    return num_correct, num_wrong, num_wrong_detected, image


def doubleTest(detected, other_detected):

    if (matchBoundingBoxes(detected, other_detected) and
         detected.label == other_detected.label and
         detected.prob < other_detected.prob):
        return True
    return False


def drawBbox(color, image, detected):
    height, width, channels = image.shape
    cv2.rectangle(image, (int(detected.bbox.xmin * width), int(detected.bbox.ymin * height)),
                  (int(detected.bbox.xmax * width), int(detected.bbox.ymax * height)), color, 2)
    image_label = '%s %f' % (detected.label, detected.prob)
    cv2.putText(image, image_label, (int(detected.bbox.xmin * width), int(detected.bbox.ymin * height)), 0, 0.6, color,
                1)

def matchBoundingBoxes(detected, annotated, max_ratio=2):

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

    if (ratio < max_ratio):
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

