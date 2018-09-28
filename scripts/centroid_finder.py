#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from creare.msg import Centroids, SolutionBounds

class text_colors:
  '''
  This function is used to produce colored output when using the print command.

  e.g. 
  print text_colors.WARNING + "Warning: Something went wrong." + text_colors.ENDCOLOR
  print text_colors.FAIL + "FAILURE: Something went wrong." + text_colors.ENDCOLOR
  print text_colors.BOLD + "Something important in bold." + text_colors.ENDCOLOR
  '''
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDCOLOR = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

class CentroidFinder(object):
  def __init__(self, show_images = False):
    self._img = None
    self._roi = [0, 0, 0, 0, False]
    self._show_images = show_images
    cam_info = rospy.wait_for_message("/camera/camera_info",CameraInfo)
    self._mtx = np.array([cam_info.K[0:3],cam_info.K[3:6],cam_info.K[6:9]])
    self._distortion = np.array(cam_info.D)

    # Image converter
    self._bridge = CvBridge()

    # Publishers
    self._pub_initial_centroids = rospy.Publisher("/centroids/initial",Centroids,queue_size=1)

    # Subscribers
    self._sub_img_raw = rospy.Subscriber("/camera/image_raw",Image,self.cbImgRaw)
    self._sub_prev_sol_bounds = rospy.Subscriber("/prev_solution_bounds",SolutionBounds,self.cbPrevSolBounds)

  def rectify(self):
    # Undistortion
    h,w = self._img.shape[:2]
    mtx_new,roi = cv2.getOptimalNewCameraMatrix(self._mtx,self._distortion,(w,h),1,(w,h))
    self._img = cv2.undistort(self._img,self._mtx,self._distortion,None,mtx_new)

  def cbPrevSolBounds(self,data):
    '''
    This callback truncates the search space if a previous solution exists.
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

    # Remove the edge of the image since there are erroneous pixels in the top left of the image (some sort of image conversion error perhaps?)
    buff = 1
    self._img = self._img[buff:self._img.shape[0] - buff, buff:self._img.shape[1] - buff]

    # Rectify the image before finding contours
    self.rectify()

    # Find centroids and reshape for ROS publishing
    centroids = self.obtain_initial_centroids()
    if len(centroids) > 0:
      num_rows = len(centroids)
      num_cols = len(centroids[0])
      centroids = centroids.reshape(1,num_rows*num_cols).tolist()[0]

    # Publish the data
    self._pub_initial_centroids.publish(centroids)

  def obtain_initial_centroids(self):
    '''
    This function processes the image and returns the centroid location in image coordinates of all contours.

    IMAGE MUST BE RECTIFIED BEFORE THIS STEP
    '''
    
    # If a valid previous solution exists, truncate image search space
    if self._roi[-1]:
      # Ensure none of region of interest boundaries are outside of image
      if self._roi[0] < 0:
        self._roi[0] = 0
      if self._roi[1] >= self._img.shape[1]:
        self._roi[1] = self._img.shape[1] - 1
      if self._roi[2] < 0:
        self._roi[2] = 0
      if self._roi[3] >= self._img.shape[0]:
        self._roi[3] = self._img.shape[0] - 1

      # Crop the image to the region of interest
      img_before = np.array(self._img)
      self._img = self._img[self._roi[2]:self._roi[3],self._roi[0]:self._roi[1]]

      if False:
        # Create a before / after image if desired
        img_to_show1 = img_before
        img_outer    = np.ones_like(img_before)
        img_outer[self._roi[2]:self._roi[3],self._roi[0]:self._roi[1]] = self._img
        img_to_show2 = img_outer
        img_to_show  = np.hstack([np.asarray(img_to_show1), np.asarray(img_to_show2)])

        cv2.imshow('reduced-search-space',img_to_show)
        cv2.waitKey(1)

    # Binarize
    max_value = 255
    block_size = 5
    const = 1
    threshold_value = 10
    _,self._img = cv2.threshold(self._img,threshold_value,max_value,cv2.THRESH_BINARY)

    if self._show_images:
      cv2.imshow('binary',self._img)
      cv2.waitKey(1)

    # Morph image to 'close' the shapes that are found
    kernel = np.ones((2,2),np.uint8)
    self._img = cv2.dilate(self._img,kernel,iterations = 1)
    self._img = cv2.erode(self._img,kernel,iterations = 1)

    if self._show_images:
      cv2.imshow('morph',self._img)
      cv2.waitKey(1)

    # Find contours
    contours, _ = cv2.findContours(self._img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # _, contours, _ = cv2.findContours(self._img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    if self._show_images:
      img_temp = np.array(self._img)
      cv2.drawContours(img_temp, contours, -1, (255,255,255), 3)
      cv2.imshow('contours',img_temp)
      cv2.waitKey(1)

    # Extract centroids
    centroids = []
    all_centers = set([])
    max_y,max_x = self._img.shape[:2]
    edge_of_img_border = 5
    near_buff = 5
    for c in contours:
      # Obtain the coordinates of the center of the contour
      (x,y),_ = cv2.minEnclosingCircle(c)
      center = (int(x),int(y))
      # If we have already detected a point at (or very near) a given coordinate, do not add it again. 
      if center not in all_centers:
        all_centers.add(center)
        # Add nearby buffer so that we do not get redundant centroids
        for i in range(-near_buff,near_buff + 1):
          for j in range(-near_buff,near_buff + 1):
            nearby_center = (center[0] + i,center[1] + j)
            all_centers.add(nearby_center)
        # Add the centroid to the collective list. Region of interest is added here since a truncated image will be offset by the amount of truncation.
        centroids.append((x + self._roi[0], y + self._roi[2]))

    if self._show_images:
      img_temp = np.array(self._img)
      for c in centroids:
        center = (int(c[0]),int(c[1]))
        cv2.circle(img_temp, center, 3, 255, 3)
      cv2.imshow('centroids',img_temp)
      cv2.waitKey(1)

    # Return the centroids found
    return np.array(centroids)

if __name__ == "__main__":
  # Initialize the node
  rospy.init_node('centroid_finder')

  # Create the node
  node = CentroidFinder(show_images = True)

  # Spin
  rospy.spin()
