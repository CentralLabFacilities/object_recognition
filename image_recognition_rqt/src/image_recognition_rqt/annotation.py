import os
import rospy
import rostopic

from qt_gui.plugin import Plugin

from python_qt_binding.QtWidgets import * 
from python_qt_binding.QtGui import * 
from python_qt_binding.QtCore import * 

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import datetime
import re
import rosservice
import time

from image_widget import ImageWidget
from dialogs import option_dialog, info_dialog, warning_dialog, number_dialog

from image_recognition_util import image_writer

_SUPPORTED_SERVICES = ["image_recognition_msgs/Annotate"]

def _sanitize(label):
    """
    Sanitize string, only allow \w regex chars
    :param label: Input that needs to be sanitized
    :return: The sanatized string
    """
    return re.sub(r'(\W+| )', '', label)


class AnnotationPlugin(Plugin):

    def __init__(self, context):
        """
        Annotation plugin to create data sets or test the Annotate.srv service
        :param context: Parent QT widget
        """
        super(AnnotationPlugin, self).__init__(context)

        # Widget setup
        self.setObjectName('Label Plugin')

        self._widget = QWidget()
        context.add_widget(self._widget)
        self._widget.resize(800,1000)
        # Layout and attach to widget
        layout = QVBoxLayout()  
        self._widget.setLayout(layout)

        self._image_widget = ImageWidget(self._widget, self._image_callback)
        layout.addWidget(self._image_widget)

        # Input field
        grid_layout = QGridLayout()
        layout.addLayout(grid_layout)

        grid_layout.addWidget(QLabel("Dilation size"), 1, 1)

        self._sliderDil = QSlider(Qt.Horizontal)
        self._sliderDil.setMinimum(1)
        self._sliderDil.setMaximum(15)
        self._sliderDil.setValue(5)
        self._sliderDil.setTickPosition(QSlider.TicksBelow)
        self._sliderDil.setTickInterval(1)

        grid_layout.addWidget(self._sliderDil, 1, 2)

        grid_layout.addWidget(QLabel("Erosion size"), 1, 3)

        self._sliderEros = QSlider(Qt.Horizontal)
        self._sliderEros.setMinimum(1)
        self._sliderEros.setMaximum(15)
        self._sliderEros.setValue(5)
        self._sliderEros.setTickPosition(QSlider.TicksBelow)
        self._sliderEros.setTickInterval(1)

        grid_layout.addWidget(self._sliderEros, 1, 4)

        self._edit_path_button = QPushButton("Edit path")
        self._edit_path_button.clicked.connect(self._get_output_directory)
        grid_layout.addWidget(self._edit_path_button, 2, 1)

        self._output_path_edit = QLineEdit()
        self._output_path_edit.setDisabled(True)
        grid_layout.addWidget(self._output_path_edit, 2, 2)


        self.labels = []
        self._option_selector = QComboBox()
        self._option_selector.currentIndexChanged.connect(self.classChange)
        grid_layout.addWidget(self._option_selector, 2, 3)

        self.classImgs = []
        self._imgNum_label = QLabel(str(0))
        grid_layout.addWidget(self._imgNum_label, 2, 4)

        self._label_edit = QLineEdit()
        grid_layout.addWidget(self._label_edit, 3, 2)

        self._edit_labels_button = QPushButton("Add Label")
        self._edit_labels_button.clicked.connect(self._add_label)
        grid_layout.addWidget(self._edit_labels_button, 3, 1)

        self._save_button = QPushButton("START")
        self._save_button.clicked.connect(self.create_dataset_clicked)
        grid_layout.addWidget(self._save_button, 3, 3)

        self._test_button = QRadioButton("TestSet")
        self._test_button.setChecked(False)
        grid_layout.addWidget(self._test_button, 3, 4)

        # Bridge for opencv conversion
        self.bridge = CvBridge()

        # Set subscriber to None
        self._sub = None
        self._srv = None

        self.interval = 3
        self.numImg = 0
        self.save = False
        self.label = ""
        self.output_directory = ""
        self.cls_id = None


    def classChange(self):
        self.label = self._option_selector.currentText()
        self.cls_id = [i[0] for i in self.labels].index(self.label)
        cls = self.labels[self.cls_id]
        self.numImg = cls[1]
        self._imgNum_label.setText(str(self.numImg))


    def create_dataset_clicked(self):
        """
        Triggered when button clicked
        """
        if not self.labels:
            warning_dialog("No labels specified!", "Please first specify some labels using the 'Edit labels' button")
            return

        if self.save:
            self._save_button.setText("START")
            self.save = False
            self.labels[self.cls_id][1] = self.numImg
        else:
            self._save_button.setText("STOP")
            self.save = True



    def annotate(self, image, bbox):
        """
        Create an annotation
        :param image: The image we want to annotate
        """
        self.annotate_srv(image, bbox)
        self.store_image(image, bbox)


    def store_image(self, image, bbox, cls_id, mask):
        """
        Store the image
        :param image: Image we would like to store
        """
        if image is not None and self.label is not None and self.output_directory is not None:
            image_writer.write_annotated(self.output_directory, image, mask, self.label, self.cls_id, bbox, self._test_button.isChecked())

    def _get_output_directory(self):
        """
        Gets and sets the output directory via a QFileDialog
        """
        self._set_output_directory(QFileDialog.getExistingDirectory(self._widget, "Select output directory"))

    def _set_output_directory(self, path):
        """
        Sets the output directory
        :param path: The path of the directory
        """
        if not path:
            path = "/tmp"

        self.output_directory = path
        self._output_path_edit.setText("Saving images to %s" % path)
        labels = self._read_labels()
        self._set_labels(labels)

    def _add_label(self):
        """
        Gets and adds a label
        """

        label = self._label_edit.text()
        labelNames = [i[0] for i in self.labels]
        if not label in list(labelNames):
            self.labels.append((label,0))
            self._option_selector.addItem(label)
            with open("{}/labels.txt".format(self.output_directory), 'a') as file:
                file.write("{}\n".format(label))
        self._label_edit.setText('')


    def _set_labels(self, labels):
        """
        Sets the labels
        :param labels: label string array
        """
        if not labels:
            labels = []
        else:
            for label in labels:
                self.labels.append(label)
                self._option_selector.addItem(label[0])

    def _read_labels(self):
        labels_tmp = []
        try:
            with open('{}/labels.txt'.format(self.output_directory), 'r') as f:
                labels_tmp = f.readlines()
            labels_tmp = [(i.rstrip(),0) for i in labels_tmp]
        except:
            pass

        labels = []
        for label in labels_tmp:
            label = list(label)
            path = "{}/{}/images".format(self.output_directory, label[0])
            if os.path.isdir(path):
                label[1] = len([name for name in os.listdir(path) if os.path.isfile("{}/{}".format(path,name))])
            labels.append(label)
        return labels

    def _image_callback(self, msg):
        """
        Called when a new sensor_msgs/Image is coming in
        :param msg: The image messaeg
        """

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            dil_size = self._sliderDil.value()
            eros_size = self._sliderEros.value()
            self.image = self._image_widget.set_image(cv_image, dil_size, eros_size)

            if self.save:
                self.numImg += 1
                self._imgNum_label.setText(str(self.numImg))
                self.store_image(self._image_widget.get_image(), self._image_widget.get_bbox(), self.cls_id, self._image_widget.get_mask())
        except CvBridgeError as e:
            rospy.logerr(e)

    def trigger_configuration(self):
        """
        Callback when the configuration button is clicked
        """
        topic_name, ok = QInputDialog.getItem(self._widget, "Select topic name", "Topic name", rostopic.find_by_type('sensor_msgs/Image'))
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
        self._widget.setWindowTitle("Label plugin, listening to (%s)" % self._sub.name)

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
        instance_settings.set_value("output_directory", self.output_directory)
        instance_settings.set_value("labels", self.labels)
        if self._sub:
            instance_settings.set_value("topic_name", self._sub.name)
    
    def _create_service_client(self, srv_name):
        """
        Create a service client proxy
        :param srv_name: Name of the service
        """
        if self._srv:
            self._srv.close()

        if srv_name in rosservice.get_service_list():
            rospy.loginfo("Creating proxy for service '%s'" % srv_name)
            self._srv = rospy.ServiceProxy(srv_name, rosservice.get_service_class_by_name(srv_name))

    def restore_settings(self, plugin_settings, instance_settings):
        """
        Callback function fired on load of the plugin that allows to restore saved variables
        :param plugin_settings: Plugin settings
        :param instance_settings: Settings of this instance
        """
        path = None
        try:
            path = instance_settings.value("output_directory")
        except:
            pass
        self._set_output_directory(path)

        labels = None
        try:
            labels = instance_settings.value("labels")
        except:
            pass
        #labels = self._read_labels()
        #self._set_labels(labels)
        self._create_service_client(str(instance_settings.value("service_name", "/image_recognition/my_service")))
        self._create_subscriber(str(instance_settings.value("topic_name", "/xtion/rgb/image_raw")))
