#!/usr/bin/env python

import cv2
import imutils
import rospy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class ImageRotator(object):
  def __init__(self):
    # Image converter
    self._bridge = CvBridge()

    # Publishers
    self._pub_rotated_img = rospy.Publisher("/camera/image_rotated",Image,queue_size=1)

    # Subscribers
    self._sub_img_raw  = rospy.Subscriber("/camera/image_raw",Image,self.cbImgRaw)

  def cbImgRaw(self,data):

    # Convert to OpenCV image
    try:
      img = self._bridge.imgmsg_to_cv2(data, 'passthrough')
    except CvBridgeError as e:
      print e

    # Rotate image without cutting anything off
    img = imutils.rotate_bound(img, 90)

    # Convert to ROS message
    try:
      img = self._bridge.cv2_to_imgmsg(img, 'mono8')
    except CvBridgeError as e:
      print(e)

    # Publish rotated image
    self._pub_rotated_img.publish(img)

if __name__ == "__main__":
  # Initialize node
  rospy.init_node('image_rotator_node')

  # Create node
  node = ImageRotator()

  # Spin
  rospy.spin()
