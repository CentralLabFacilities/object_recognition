from python_qt_binding.QtWidgets import * 
from python_qt_binding.QtGui import * 
from python_qt_binding.QtCore import * 
import cv2
import numpy as np

def _convert_cv_to_qt_image(cv_image):
    """
    Method to convert an opencv image to a QT image
    :param cv_image: The opencv image
    :return: The QT Image
    """
    cv_image = cv_image.copy() # Create a copy
    height, width, byte_value = cv_image.shape
    byte_value = byte_value * width
    cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB, cv_image)

    return QImage(cv_image, width, height, byte_value, QImage.Format_RGB888)


class ImageWidget(QWidget):

    def __init__(self, parent, image_callback):
        """
        Image widget that allows drawing rectangles and firing a image_roi_callback
        :param parent: The parent QT Widget
        :param image_callback: The callback function when a ROI is drawn
        """
        super(ImageWidget, self).__init__(parent)
        self._cv_image = None
        self._qt_image = QImage()

        self.image_callback = image_callback

        self.bbox = None
        self.pMOG2 = cv2.createBackgroundSubtractorMOG2(500,16,True)

        self.erosion_size = 5
        self.dilation_size = 5
        self.detections = []

    def get_image(self):
        # Flip if we have dragged the other way
        return self._cv_image

    def set_image(self, image):
        """
        Sets an opencv image to the widget
        :param image: The opencv image
        """
        self._cv_image = image
        self._qt_image = _convert_cv_to_qt_image(image)
        self.update()


    def calc_bbox(self, dil_num):
        self.dilation_size = dil_num

        fgMaskMOG2 = None
        self.pMOG2.apply(self._cv_image, fgMaskMOG2, 0.001)
        cv2.inRange(fgMaskMOG2, cv2.Scalar(250,250,250), cv2.Scalar(255,255,255), fgMaskMOG2)
        mask = cv2.Mat(fgMaskMOG2.size(), fgMaskMOG2.type())
        mask.setTo(cv2.Scalar(0,0,0))
        cv2.rectangle(mask, cv2.Rect(150,110,300,330), cv2.Scalar(255,255,255), cv2.FILLED)
        maskedImage = None
        fgMaskMOG2.copyTo(maskedImage, mask)
        maskedImage.copyTo(fgMaskMOG2)

        elementEr = cv2.getStructuringElement(cv2.MORPH_RECT, (self.erosion_size, self.erosion_size), (-1,-1))
        cv2.dilate(fgMaskMOG2, fgMaskMOG2, elementEr)

        elementDi = cv2.getStructuringElement(cv2.MORPH_RECT, (self.erosion_size, self.erosion_size), (-1,-1))
        cv2.erode(fgMaskMOG2, fgMaskMOG2, elementDi)

        diagElem = np.identity(10)
        cv2.erode(fgMaskMOG2, fgMaskMOG2, diagElem)
        cv2.dilate(fgMaskMOG2, fgMaskMOG2, diagElem)

        diagElem2 = np.flip(diagElem, 1)
        cv2.erode(fgMaskMOG2, fgMaskMOG2, diagElem2)
        cv2.dilate(fgMaskMOG2, fgMaskMOG2, diagElem2)

        shapeHeight = 2
        shapeWidth = 5

        for i in range(2):
            elementEr = cv2.getStructuringElement(cv2.MORPH_RECT, (shapeHeight, shapeWidth), (-1,-1))
            cv2.erode(fgMaskMOG2, fgMaskMOG2, elementEr)
            elementDi = cv2.getStructuringElement(cv2.MORPH_RECT, (shapeHeight, shapeWidth), (-1,-1))
            cv2.dilate(fgMaskMOG2, fgMaskMOG2, elementDi)

        thresh = 1
        cv2.blur(fgMaskMOG2, fgMaskMOG2, (6,6))
        cv2.Canny(fgMaskMOG2, fgMaskMOG2, thresh, thresh*2, 3)

        dil_size = 2
        elementDi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dil_size+1, 2*dil_size+1), (dil_size, dil_size))
        cv2.dilate(fgMaskMOG2, fgMaskMOG2, elementDi)

        contours = None
        hierarchy = None

        cv2.findContours(fgMaskMOG2, contours, hierarchy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, (0,0))

        largest_area = 0
        largest_contour_index = 0

        for i in range(len(contours)):
            a = cv2.contourArea(contours[i], False)
            if a > largest_area:
                largest_area = a
                largest_contour_index = i

        cv2.convexHull(contours[largest_contour_index], contours[largest_contour_index])
        self.bbox = cv2.boundingRect(contours[largest_contour_index])

        cv2.rectangle(self._cv_image, self.bbox, (0,0,255))


    def get_bbox(self):
        return self.bbox