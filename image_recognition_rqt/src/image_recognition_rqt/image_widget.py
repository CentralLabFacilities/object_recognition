from python_qt_binding.QtWidgets import * 
from python_qt_binding.QtGui import * 
from python_qt_binding.QtCore import * 
import cv2
import numpy as np
import copy

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
	self._bg_image = None
        self._qt_image = QImage()
        self.pMOG2 = cv2.createBackgroundSubtractorMOG2(500, 16, True)

        self.clip_rect = QRect(0, 0, 0, 0)

        self.dragging = False
        self.drag_offset = QPoint()

        self.detections = []
        self.bbox = None
	self.mask = None


    def paintEvent(self, event):
        """
        Called every tick, paint event of QT
        :param event: Paint event of QT
        """
        painter = QPainter()
        painter.begin(self)
        painter.drawImage(0, 0, self._qt_image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.cyan, 5.0))
        painter.drawRect(self.clip_rect)

        painter.setFont(QFont('Decorative', 10))
        for rect, label in self.detections:
            painter.setPen(QPen(Qt.magenta, 5.0))
            painter.drawRect(rect)




        #self.image_callback = image_callback

    def calc_bbox(self, image, dil_size, eros_size):
        fgMaskMOG2 = self.pMOG2.apply(image, 0.001)
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


	#copy image to different background
	back = cv2.imread('backgrounds/rgb-1577.ppm',1)
	mask1 = fgMaskMOG2

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

            hull = cv2.convexHull(contours[largest_contour_index], contours[largest_contour_index])
            mask_poly = np.zeros(fgMaskMOG2.shape, dtype=np.uint8)
	    cv2.fillConvexPoly(mask_poly, hull.astype(np.int32), (255))
	    #cv2.imshow('mask test',mask_poly)

            bbox = cv2.boundingRect(contours[largest_contour_index])
            mask = np.zeros(fgMaskMOG2.shape, dtype=np.uint8)
            rect = cv2.minAreaRect(contours[largest_contour_index])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), 255, thickness=-1)
	    mask1 = cv2.bitwise_and(mask_poly, mask_poly, mask = mask)
	    self.mask = mask1
	    #cv2.imshow('mask_res',mask1)
            fg = cv2.bitwise_and(image,image, mask = mask1)
            mask1_inv = cv2.bitwise_not(mask1)
            bg = cv2.bitwise_and(back,back, mask = mask1_inv)
            result = bg + fg
	    #cv2.imshow('img', result)
	    #image = result
	    #self._bg_image = image
	    self.bbox = cv2.boundingRect(contours[largest_contour_index])
	    self.bbox = (self.bbox[0], self.bbox[0] + self.bbox[2], self.bbox[1], self.bbox[1] + self.bbox[3])
	    cv2.rectangle(image, (self.bbox[0], self.bbox[2]), (self.bbox[1], self.bbox[3]), (0, 0, 255))

        return image

    def get_image(self):
        return self._cv_image

    def get_bg_image(self):
        return self._bg_image

    def get_mask(self):
        return self.mask

    def set_image(self, image, dil_size, eros_size):
        """
        Sets an opencv image to the widget
        :param image: The opencv image
        """
        self._cv_image = copy.copy(image)
        image = self.calc_bbox(image, dil_size, eros_size)
        self._qt_image = _convert_cv_to_qt_image(image)
        self.update()
	return image

    def get_mask(self):
	return self.mask

    def get_bbox(self):
        return self.bbox
