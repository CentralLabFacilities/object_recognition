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


def _get_roi_from_rect(rect):
    """
    Returns the ROI from a rectangle, the rectangle can have the top and bottom flipped
    :param rect: Rect to get roi from
    :return: x, y, width, height of ROI
    """
    x_min = min(rect.topLeft().x(), rect.bottomRight().x())
    y_min = min(rect.topLeft().y(), rect.bottomRight().y())
    x_max = max(rect.topLeft().x(), rect.bottomRight().x())
    y_max = max(rect.topLeft().y(), rect.bottomRight().y())

    return x_min, y_min, x_max - x_min, y_max - y_min



class ImageWidget(QWidget):

    

    def __init__(self, parent, image_callback, clear_on_click=False):
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
        self.image_callback = image_callback

        self.detections = []
        self._clear_on_click = clear_on_click
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

	    painter.setPen(QPen(Qt.magenta, 5.0))
            painter.drawText(rect, Qt.AlignCenter, label)

        painter.end()


        

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

    def get_roi_image(self):
        # Flip if we have dragged the other way
        x, y, width, height = _get_roi_from_rect(self.clip_rect)
	self.bbox = (x, x+width, y, y+height)

        return self._cv_image[y:y + height, x:x + width]

    def set_image(self, img, bboxes, labels):
        """
        Sets an opencv image to the widget
        :param image: The opencv image
        """
        image = img
        self._cv_image = copy.copy(image)

        if (bboxes is not None and labels is not None):
            # draw boxes for current annotations
            for i in range (0,len(bboxes)):
                bbox = bboxes[i]
                label = labels[i]
                color = (0, 0, 255)
                cv2.rectangle(image, (bbox[0], bbox[2]),
                          (bbox[1], bbox[3]), color, 2)
                image_label = '%s' % (label)
                cv2.putText(image, image_label, (bbox[0], bbox[2]), 0, 0.6,
                        color,
                        1)
        self._qt_image = _convert_cv_to_qt_image(image)
        self.update()
	return self._cv_image

    def get_mask(self):
	return self.mask

    def get_bbox(self):
        return self.bbox

    def add_detection(self, x, y, width, height, label):
        """
        Adds a detection to the image
        :param x: ROI_X
        :param y: ROI_Y
        :param width: ROI_WIDTH
        :param height: ROI_HEIGHT
        :param label: Text to draw
        """
        roi_x, roi_y, roi_width, roi_height = _get_roi_from_rect(self.clip_rect)
        self.detections.append((QRect(x+roi_x, y+roi_y, width, height), label))

    def clear(self):
        self.detections = []
        self.clip_rect = QRect(0, 0, 0, 0)

    def get_roi(self):
        return _get_roi_from_rect(self.clip_rect)

    def mousePressEvent(self, event):
        """
        Mouspress callback
        :param event: mouse event
        """
        # Check if we clicked on the img
        if event.pos().x() < self._qt_image.width() and event.pos().y() < self._qt_image.height():
            if self._clear_on_click:
                self.clear()
            self.clip_rect.setTopLeft(event.pos())
            self.clip_rect.setBottomRight(event.pos())
            self.dragging = True

    def mouseMoveEvent(self, event):
        """
        Mousemove event
        :param event: mouse event
        """
        if not self.dragging:
            return

        self.clip_rect.setBottomRight(event.pos())
        
        self.update()
    
    def mouseReleaseEvent(self, event):
        """
        Mouse release event
        :param event: mouse event
        """
        if not self.dragging:
            return

        roi_image = self.get_roi_image()
        if roi_image is not None:
            self.image_callback(roi_image)

        self.dragging = False
