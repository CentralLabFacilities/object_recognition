import os
import sys

import rospy
import rostopic
import rosservice

from qt_gui.plugin import Plugin

from python_qt_binding.QtWidgets import * 
from python_qt_binding.QtGui import * 
from python_qt_binding.QtCore import * 

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from image_widget_test_gui import ImageWidget
from dialogs import option_dialog, warning_dialog

from object_tracking_msgs.srv import Classify2D
from vision_msgs.msg import ObjectHypothesis, Classification2D
_SUPPORTED_SERVICES = ["object_tracking_msgs/Classify2D"]


class TestPlugin(Plugin):

    def __init__(self, context):
        """
        TestPlugin class to evaluate the image_recognition_msgs interfaces
        :param context: QT context, aka parent
        """
        super(TestPlugin, self).__init__(context)

        # Widget setup
        self.setObjectName('Test Plugin')

        self._widget = QWidget()
        context.add_widget(self._widget)
        
        # Layout and attach to widget
        layout = QVBoxLayout()  
        self._widget.setLayout(layout)

        self._image_widget = ImageWidget(self._widget, self.image_roi_callback, clear_on_click=True)
        layout.addWidget(self._image_widget)

        # Input field
        grid_layout = QGridLayout()
        layout.addLayout(grid_layout)

        self._info = QLineEdit()
        self._info.setDisabled(True)
        self._info.setText("Draw a rectangle on the screen to perform recognition of that ROI")
        layout.addWidget(self._info)

        # Bridge for opencv conversion
        self.bridge = CvBridge()

        # Set subscriber and service to None
        self._sub = None
        self._srv = None

        self._unknown_probability = 0.1

        self.label_path = ""

        self._edit_label_path_button = QPushButton("Edit label path")
        self._edit_label_path_button.clicked.connect(self._get_label_path)
        layout.addWidget(self._edit_label_path_button)

        self._label_path_edit = QLineEdit()
        self._label_path_edit.setDisabled(True)
        layout.addWidget(self._label_path_edit)


    def _set_label_path(self, path):
        """
        Set the image directory
        :param path: image dir
        """
        if not path:
            path = ""
        else:
            path = str(path).split("\'")[1]

        self.label_path = path
        self._label_path_edit.setText("Read labels from %s" % path)
        self.read_labels()

    def _get_label_path(self):
        """
        Get and set image directory with use of QFileDialog GUI
        """
        self._set_label_path(QFileDialog.getOpenFileName(self._widget, "Select images directory", "/", "Text files (*.txt)"))

    def classify_srv_call(self, roi_image):
        """
        Method that calls the Classify2D.srv
        :param roi_image: Selected roi_image by the user
        """
        try:
            images = []
            image = self.bridge.cv2_to_imgmsg(roi_image, "bgr8")
            images.append(image)
            result = self._srv(images)
        except Exception as e:
            warning_dialog("Service Exception", str(e))
            return

        # we send one image, so we get max. one result
        if len(result.classifications) == 0:
            return

        c = result.classifications[0]
        text_array = []
        best = ObjectHypothesis(id=0, score=self._unknown_probability) # 0 -> unknown

        for r in c.results:
            text_array.append("%s: %.2f" % (self.labels[r.id], r.score))
            if r.score > best.score:
                best = r

        self._image_widget.add_detection(0, 0, 1, 1, str(best.id))

        if text_array:
            option_dialog("Classification results (Unknown probability=%.2f)" %
                          self._unknown_probability,
                          text_array)  # Show all results in a dropdown


    def image_roi_callback(self, roi_image):
        """
        Callback triggered when the user has drawn an ROI on the image
        :param roi_image: The opencv image in the ROI
        """
        if self._srv is None:
            warning_dialog("No service specified!",
                           "Please first specify a service via the options button (top-right gear wheel)")
            return

        if self.label_path == "":
            warning_dialog("No object labels defined!",
                           "Please load a label file first.")
            return

        if self._srv.service_class == Classify2D:
            self.classify_srv_call(roi_image)
        else:
            warning_dialog("Unknown service class", "Service class is unkown!")

    def _image_callback(self, msg):
        """
        Sensor_msgs/Image callback
        :param msg: The image message
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

        self._image_widget.set_image(cv_image)

    def read_labels(self):
        with open(self.label_path, 'rb') as f:
            self.labels = f.read().split("\n")

    def trigger_configuration(self):
        """
        Callback when the configuration button is clicked
        """
        topic_name, ok = QInputDialog.getItem(self._widget, "Select topic name", "Topic name",
                                              rostopic.find_by_type('sensor_msgs/Image'))
        if ok:
            self._create_subscriber(topic_name)

        available_rosservices = []
        for s in rosservice.get_service_list():
            try:
                if rosservice.get_service_type(s) in _SUPPORTED_SERVICES:
                    available_rosservices.append(s)
            except:
                pass

        srv_name, ok = QInputDialog.getItem(self._widget, "Select service name", "Service name", available_rosservices)
        if ok:
            self._create_service_client(srv_name)

    def _create_subscriber(self, topic_name):
        """
        Method that creates a subscriber to a sensor_msgs/Image topic
        :param topic_name: The topic_name
        """
        if self._sub:
            self._sub.unregister()
        self._sub = rospy.Subscriber(topic_name, Image, self._image_callback)
        rospy.loginfo("Listening to %s -- spinning .." % self._sub.name)
        self._widget.setWindowTitle("Test plugin, listening to (%s)" % self._sub.name)

    def _create_service_client(self, srv_name):
        """
        Method that creates a client service proxy to call either the GetFaceProperties.srv or the Classify2D.srv
        :param srv_name:
        """
        if self._srv:
            self._srv.close()

        if srv_name in rosservice.get_service_list():
            rospy.loginfo("Creating proxy for service '%s'" % srv_name)
            self._srv = rospy.ServiceProxy(srv_name, rosservice.get_service_class_by_name(srv_name))

    def shutdown_plugin(self):
        """
        Callback function when shutdown is requested
        """
        pass

    def save_settings(self, plugin_settings, instance_settings):
        """
        Callback function on shutdown to store the local plugin variables
        :param plugin_settings: Plugin settings
        :param instance_settings: Settings of this instance
        """
        if self._sub:
            instance_settings.set_value("topic_name", self._sub.name)

    def restore_settings(self, plugin_settings, instance_settings):
        """
        Callback function fired on load of the plugin that allows to restore saved variables
        :param plugin_settings: Plugin settings
        :param instance_settings: Settings of this instance
        """
        self._create_subscriber(str(instance_settings.value("topic_name", "/xtion/rgb/image_raw")))
        self._create_service_client(str(instance_settings.value("service_name", "/image_recognition/my_service")))
