cmake_minimum_required(VERSION 2.8.3)
project(tensorflow_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  object_tracking_msgs
  vision_msgs
  image_recognition_util
)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/retrain
  scripts/object_recognition_node
  scripts/segment_classify_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
