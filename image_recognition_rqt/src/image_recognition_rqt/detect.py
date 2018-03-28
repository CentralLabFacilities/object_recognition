import rospy
import rostopic
import rosservice
import cv2

from qt_gui.plugin import Plugin

from python_qt_binding.QtWidgets import *
from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from object_tracking_msgs.msg import ObjectLocation, Hypothesis

from image_widget import ImageWidget
from dialogs import option_dialog, warning_dialog, info_dialog

from object_tracking_msgs.srv import DetectObjects

_SUPPORTED_SERVICES = ["object_tracking_msgs/DetectObjects"]

from tf_detector import TfDetector


class DetectPlugin(Plugin):
    def __init__(self, context):
        """
        DetectPlugin class to evaluate the object_tracking_msgs interfaces
        :param context: QT context, aka parent
        """
        super(DetectPlugin, self).__init__(context)

        # Widget setup
        self.setObjectName('Detection Plugin')

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
        self._info.setText("Detect objects with tensorflow")
        layout.addWidget(self._info)

        # Bridge for opencv conversion
        self.bridge = CvBridge()

        # Set subscriber and service to None
        self._sub = None
        self._srv = None
        self._srv_depthLookup = None
        self._srv_segment = None

        self.tf_detector = TfDetector()
        self.cv_image = None

    # TODO
    def detect_srv_call(self, image):
        """
        Method that calls the DetectObjects.srv
        :param roi_image: Selected roi_image by the user
        """
        #imageReq = self.bridge.cv2_to_imgmsg(image, "bgr8")
        imageReq = None
        try:
            result = self._srv(imageReq)
        except Exception as e:
            warning_dialog("Service Exception", str(e))
            return
        #Call depth lookup and segmentation
        if (self._srv_depthLookup and self._srv_segment):
            self.triggerSegmentation(result.objectLocationList)
        self.visualize_bounding_boxes(result.objectLocationList, image)


    def triggerSegmentation(self,detectionResult):

        print("do depth lookup")
        depthLookupResult = self._srv_depthLookup(detectionResult)

        #todo: service call to segmentation with result of depthLookup (use 1 objectShape)
        print depthLookupResult
        print("do segmentation for each objectShape")
        self._srv_segment(depthLookupResult.objectShapeList)

    # TODO: replace with button callback
    def image_roi_callback(self, roi_image):
        """
        Callback triggered when the user has drawn an ROI on the image
        :param roi_image: The opencv image in the ROI
        """
        if self._srv is None:
            warning_dialog("No service specified!",
                           "Please first specify a service via the options button (top-right gear wheel)")
            return

        if self._srv.service_class == DetectObjects:
            self.detect_srv_call(self.cv_image)
        else:
            warning_dialog("Unknown service class", "Service class is unkown!")

    def visualize_bounding_boxes(self, result, image):
        # visualization of detection results
        for i in range(0, len(result)):
            prob = result[i].hypotheses[0].reliability
            label = result[i].hypotheses[0].label
            bbox = result[i].bounding_box
            cv2.rectangle(image, (int(bbox.x_offset), int(bbox.y_offset)),
                          (int(bbox.x_offset + bbox.width), int(bbox.y_offset + bbox.height)), (0, 100, 200), 2)
            image_label = '%s %f' % (label, prob)
            cv2.putText(image, image_label, (int(bbox.x_offset), int(bbox.y_offset)), 0, 0.4, (0, 0, 255), 1)
        cv2.imshow('image', image)
        cv2.waitKey(0)

    def _image_callback(self, msg):
        """
        Sensor_msgs/Image callback
        :param msg: The image message
        """
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

        # update image widget
        self._image_widget.set_image(self.cv_image, None, None)

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
        Method that creates a client service proxy to call either the GetFaceProperties.srv or the DetectObjects.srv
        :param srv_name:
        """
        if self._srv:
            self._srv.close()
        if srv_name in rosservice.get_service_list():
            rospy.loginfo("Creating proxy for service '%s'" % srv_name)
            self._srv = rospy.ServiceProxy(srv_name, rosservice.get_service_class_by_name(srv_name))
        else:
            rospy.loginfo("Service client with name '%s' cannot be created" % srv_name)
        #set depthLookup service
        if not self._srv_depthLookup:
            dl_name = "/clf_perception_depth_lookup_objects/depthLookup"
            if dl_name in rosservice.get_service_list():
                self._srv_depthLookup = rospy.ServiceProxy(dl_name, rosservice.get_service_class_by_name(dl_name))
            else:
                rospy.loginfo("Service client with name '%s' cannot be created" % dl_name)
        if not self._srv_segment:
            seg_name = "/segment"
            if seg_name in rosservice.get_service_list():
                self._srv_segment = rospy.ServiceProxy(seg_name, rosservice.get_service_class_by_name(seg_name))
            else:
                rospy.loginfo("Service client with name '%s' cannot be created" % seg_name)

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
        self._create_service_client(str(instance_settings.value("service_name", "/detect")))
