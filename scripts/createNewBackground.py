import cv2
import numpy as np
import sys
eros_size = 5
dil_size = 5

if len(sys.argv) < 3:
    print('usage: python createNewBackground.py <videofile> <background Image>')

cap = cv2.VideoCapture(sys.argv[1])
bgimg = cv2.imread(sys.argv[2])
pMOG2 = cv2.createBackgroundSubtractorMOG2(500, 16, True)
while True:

    ret, image = cap.read()
    bgimg = cv2.resize(bgimg, (640,480))

    fgMaskMOG2 = pMOG2.apply(image, 0.001)
    fgMaskMOG2 = cv2.inRange(fgMaskMOG2, 250, 255)

    elementEr = cv2.getStructuringElement(cv2.MORPH_RECT, (eros_size, eros_size), (-1, -1))
    fgMaskMOG2 = cv2.dilate(fgMaskMOG2, elementEr)

    elementDi = cv2.getStructuringElement(cv2.MORPH_RECT, (dil_size, dil_size), (-1, -1))
    fgMaskMOG2 = cv2.erode(fgMaskMOG2, elementDi)

    diagElem = np.identity(10, np.uint8)
    fgMaskMOG2 = cv2.erode(fgMaskMOG2, diagElem)
    fgMaskMOG2 = cv2.dilate(fgMaskMOG2, diagElem)

    diagElem2 = np.fliplr(diagElem)
    fgMaskMOG2 = cv2.erode(fgMaskMOG2, diagElem2)
    fgMaskMOG2 = cv2.dilate(fgMaskMOG2, diagElem2)

    shapeHeight = 2
    shapeWidth = 5

    elementEr = cv2.getStructuringElement(cv2.MORPH_RECT, (shapeHeight, shapeWidth), (-1, -1))
    fgMaskMOG2 = cv2.erode(fgMaskMOG2, elementEr)
    elementDi = cv2.getStructuringElement(cv2.MORPH_RECT, (shapeHeight, shapeWidth), (-1, -1))
    fgMaskMOG2 = cv2.dilate(fgMaskMOG2, elementDi)
    elementEr = cv2.getStructuringElement(cv2.MORPH_RECT, (shapeWidth, shapeHeight), (-1, -1))
    fgMaskMOG2 = cv2.erode(fgMaskMOG2, elementEr)
    elementDi = cv2.getStructuringElement(cv2.MORPH_RECT, (shapeWidth, shapeHeight), (-1, -1))
    fgMaskMOG2 = cv2.dilate(fgMaskMOG2, elementDi)


    thresh = 1
    fgMaskMOG2 = cv2.blur(fgMaskMOG2, (6, 6))
    fgMaskMOG2 = cv2.Canny(fgMaskMOG2, thresh, thresh * 2, 3)
    dil_size = 4
    elementDi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil_size + 1, 2 * dil_size + 1),
                                          (dil_size, dil_size))
    fgMaskMOG2 = cv2.dilate(fgMaskMOG2, elementDi)

    fgMaskMOG2, contours, hierarchy = cv2.findContours(fgMaskMOG2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest_area = 0
    largest_contour_index = 0

    for i in range(len(contours)):
        a = cv2.contourArea(contours[i], False)
        if a > largest_area:
            largest_area = a
            largest_contour_index = i
    if len(contours) > 0:
        res = cv2.bitwise_and(image, image, mask = fgMaskMOG2)
        cv2.convexHull(contours[largest_contour_index], contours[largest_contour_index])
        bbox = cv2.boundingRect(contours[largest_contour_index])
        mask = np.zeros(fgMaskMOG2.shape, dtype=np.uint8)
        rect = cv2.minAreaRect(contours[largest_contour_index])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), 255, thickness=-1)
        fg = cv2.bitwise_and(image,image, mask = mask)
        mask = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(bgimg,bgimg, mask = mask)
        result = cv2.add(bg,fg)
        cv2.imshow('img', result)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()