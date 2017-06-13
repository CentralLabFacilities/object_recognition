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
    print(width)
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




	self.clip_rect = QRect(0, 0, 0, 0)
        self.dragging = False
        self.drag_offset = QPoint()
        #self.image_roi_callback = image_roi_callback

        self.detections = []
        #self._clear_on_click = clear_on_click

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
	#print(self._qt_image)
        self.update()


    def calc_bbox(self, dil_num):
        self.dilation_size = dil_num

        pMOG2 = cv2.createBackgroundSubtractorMOG2(500, 16, True)
        fgMaskMOG2 = pMOG2.apply(self._cv_image, 0.001)
        cv2.inRange(fgMaskMOG2, 250, 255)
        mask = np.zeros(fgMaskMOG2.shape, np.uint8)
        mask = cv2.rectangle(mask, (150, 110), (300, 330), (255, 255, 255), cv2.FILLED)
        fgMaskMOG2 = cv2.bitwise_and(fgMaskMOG2, fgMaskMOG2, mask=mask)

        erosion_size = 5

        elementEr = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_size, erosion_size), (-1, -1))
        fgMaskMOG2 = cv2.dilate(fgMaskMOG2, elementEr)

        elementDi = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_size, erosion_size), (-1, -1))
        fgMaskMOG2 = cv2.erode(fgMaskMOG2, elementDi)

        diagElem = np.identity(10, np.uint8)
        fgMaskMOG2 = cv2.erode(fgMaskMOG2, diagElem)
        fgMaskMOG2 = cv2.dilate(fgMaskMOG2, diagElem)

#        diagElem2 = np.flip(diagElem, 1)
        diagElem2 = np.fliplr(diagElem)
        fgMaskMOG2 = cv2.erode(fgMaskMOG2, diagElem2)
        fgMaskMOG2 = cv2.dilate(fgMaskMOG2, diagElem2)

        shapeHeight = 2
        shapeWidth = 5

        for i in range(2):
            elementEr = cv2.getStructuringElement(cv2.MORPH_RECT, (shapeHeight, shapeWidth), (-1, -1))
            fgMaskMOG2 = cv2.erode(fgMaskMOG2, elementEr)
            elementDi = cv2.getStructuringElement(cv2.MORPH_RECT, (shapeHeight, shapeWidth), (-1, -1))
            fgMaskMOG2 = cv2.dilate(fgMaskMOG2, elementDi)

        thresh = 1
        fgMaskMOG2 = cv2.blur(fgMaskMOG2, (6, 6))
        fgMaskMOG2 = cv2.Canny(fgMaskMOG2, thresh, thresh * 2, 3)

        dil_size = 2
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

        cv2.convexHull(contours[largest_contour_index], contours[largest_contour_index])
        self.bbox = cv2.boundingRect(contours[largest_contour_index])
        self.bbox = (self.bbox[0], self.bbox[0]+ self.bbox[2], self.bbox[1], self.bbox[1] + self.bbox[3])
        cv2.rectangle(self._cv_image, (self.bbox[0], self.bbox[2]), (self.bbox[1], self.bbox[3]), (0, 0, 255))
	#print(self.bbox[0])

    def get_bbox(self):
        return self.bbox
