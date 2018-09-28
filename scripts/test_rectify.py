#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from creare.msg import Centroids, SolutionBounds

class ImageProcessor(object):
  def __init__(self, show_images = False):
    self._img = None
    self._roi = [0, 0, 0, 0, False]
    cam_info = rospy.wait_for_message("/camera/camera_info",CameraInfo)
    self._mtx = np.array([cam_info.K[0:3],cam_info.K[3:6],cam_info.K[6:9]])
    self._distortion = np.array(cam_info.D)

    # Image converter
    self._bridge = CvBridge()

    # Subscribers
    self.sub_img_raw = rospy.Subscriber("/camera/image_raw",Image,self.cbImgRaw)
    self.sub_prev_sol_bounds = rospy.Subscriber("/prev_solution_bounds",SolutionBounds,self.cbPrevSolBounds)

  def rectify(self):
    # Undistortion
    h,w = self._img.shape[:2]
    mtx_new,roi = cv2.getOptimalNewCameraMatrix(self._mtx,self._distortion,(w,h),1,(w,h))
    self._img = cv2.undistort(self._img,self._mtx,self._distortion,None,mtx_new)
    # # Cropping
    # x,y,w,h = roi
    # self._img = self._img[y:y+h, x:x+w]    

  def cbPrevSolBounds(self,data):
    '''
    This callback sets a region of interest if a previous solution exists.
    '''

    if data.sol_exists:
      # Define the boundaries of the region of interest
      buff = 150
      self._roi = [data.left - buff, data.right + buff, data.top - buff, data.bottom + buff, True]
      self._roi[0:4] = [int(val) for val in self._roi[0:4]]
    else:
      self._roi = [0, 0, 0, 0, False]

  def cbImgRaw(self,data):
    '''
    This callback finds and publishes all contours in an image.
    '''

    # Convert the ROS image to an OpenCV image
    try:
      self._img = self._bridge.imgmsg_to_cv2(data, '8UC1')
    except CvBridgeError as e:
      print text_colors.WARNING + 'Warning: Image converted unsuccessfully before processing.' + text_colors.ENDCOLOR

    cv2.imshow('orig',self._img)
    cv2.waitKey(1)

    # Rectify the image before finding contours
    self.rectify()

    cv2.imshow('rectified',self._img)
    cv2.waitKey(1)

if __name__ == "__main__":
  # Initialize the node
  rospy.init_node('test_rectify')

  # Create the node
  node = ImageProcessor(show_images = True)

  # Spin
  rospy.spin()
