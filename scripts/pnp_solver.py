#!/usr/bin/env python

import rospy
import numpy as np
import math
import cv2
from sensor_msgs.msg import CameraInfo
from itertools import combinations
from creare.msg import Centroids, PoseEstimate, SolutionBounds

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

def rot2euler(R): 
  sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
  singular = sy < 1e-6
  if  not singular :
      x = math.atan2(R[2,1] , R[2,2])
      y = math.atan2(-R[2,0], sy)
      z = math.atan2(R[1,0], R[0,0])
  else :
      x = math.atan2(-R[1,2], R[1,1])
      y = math.atan2(-R[2,0], sy)
      z = 0
  return np.array([x, y, z])

def assign_points(centroids):
  '''
  This function assigns image points (centroids) to their corresponding real-world coordinate based upon a linear regression of subsets of four points. The feature has two rows of 4 linear points, so we expect two subsets of four points with relatively low residuals.

  The length of centroids passed in must be exactly 8 or 'other_row' will not have the correct number of points.
  '''

  centroids = [list(c) for c in centroids]

  if len(centroids) == 8:
    # Find the first low-residual subset
    subsets = combinations(centroids,4)
    rows = []
    k = 0
    for s in subsets:
      if len(rows) < 1:
        x = [p[0] for p in s]
        y = [p[1] for p in s]
        _, residuals, _, _, _ = np.polyfit(x, y, 1, full = True)
        if residuals < 1:
          rows.append(list(s))
        k = k + 1

    # If no subset seems to be linear, return empty
    if len(rows) < 1:
      return [[0],[0]]

    # Now that we have assigned one row, we can deduce the other row. Eliminate the determined row from the list of centroids. The remaining 4 points are the other row. 
    other_row = [x for x in centroids if x not in rows[0]]
    rows.append(list(other_row))

    # Now we have both rows, so we must decide which is the top row and which is the bottom row. First, sort each row so that the points in each row are organized from right to left in the image.
    for r in rows:
      r.sort(key=lambda x: x[0])

    # Then, use the first element of each row to determine which row is on top
    if rows[0][0][1] < rows[1][0][1]:
      top_row    = rows[0]
      bottom_row = rows[1]
    else:
      top_row    = rows[1]
      bottom_row = rows[0]

    return [bottom_row,top_row]
  else:
    return [[0],[0]]

def pose_extraction(centroids, use_prev_solution, prev_rvecs, prev_tvecs, mtx, distortion):
  # Assign centroids to real-world points
  assigned_points = assign_points(centroids)
  total_points_assigned = len(assigned_points[0]) + len(assigned_points[1])
  if not total_points_assigned == 8:
    print text_colors.FAIL + "Failure: Failure to assign points correctly." + text_colors.ENDCOLOR
    return None

  # Calculate pose. First, define object points. The units used here, [cm], will determine the units of the output. These are the relative positions of the beacons in NED GCS-frame coordinates (aft, port, down).
  objp = np.zeros((8,1,3), np.float32)
  # Currently set to vicon feature
  row_aft = [0,-0.802] # [m]
  row_port = [[0.0, -0.161, -0.318, -0.476],[0.0, -0.159, -0.318, -0.472]]
  row_down = [0,-0.256] # [m]
  # Lower row of beacons
  objp[0] = [ row_aft[0], row_port[0][0], row_down[0]]
  objp[1] = [ row_aft[0], row_port[0][1], row_down[0]]
  objp[2] = [ row_aft[0], row_port[0][2], row_down[0]]
  objp[3] = [ row_aft[0], row_port[0][3], row_down[0]]
  # Upper row of beacons
  objp[4] = [ row_aft[1], row_port[1][0], row_down[1]]
  objp[5] = [ row_aft[1], row_port[1][1], row_down[1]]
  objp[6] = [ row_aft[1], row_port[1][2], row_down[1]]
  objp[7] = [ row_aft[1], row_port[1][3], row_down[1]]

  # Define feature points by the correspondences determined above. The bottom row (assigned_points[0]) corresponds to the lower row of beacons, and the top row (assigned_points[1]) corresponds to the upper row. Each row in assigned_points is arranged with the leftmost point in the image in the first index, and so on. 
  feature_points = np.zeros((8,1,2), np.float32)
  # Lowermost Subset
  feature_points[0] = assigned_points[0][0]
  feature_points[1] = assigned_points[0][1]
  feature_points[2] = assigned_points[0][2]
  feature_points[3] = assigned_points[0][3]
  # Uppermost Subset
  feature_points[4] = assigned_points[1][0]
  feature_points[5] = assigned_points[1][1]
  feature_points[6] = assigned_points[1][2]
  feature_points[7] = assigned_points[1][3]

  # Define the axis of the feature
  axis = np.float32([[0.5,0,0], [0,0.5,0], [0,0,0.5]]).reshape(-1,3)

  # Find rotation and translation vectors
  flag_success,rvecs,tvecs = cv2.solvePnP(objp,feature_points,mtx,distortion,prev_rvecs,prev_tvecs,use_prev_solution,cv2.CV_ITERATIVE)
  
  if flag_success:
    # Calculate pose
    Pc = tuple(feature_points[0].ravel())
    Pc = np.array([[Pc[0]], [Pc[1]], [1]])
    Kinv = np.matrix(np.linalg.inv(mtx))
    R,_ = cv2.Rodrigues(rvecs)
    Rinv = np.matrix(np.linalg.inv(R))
    T = np.array(tvecs)
    position = Rinv*(Kinv*Pc-T)

    orientation = rot2euler(R)
    # Return the obtained pose, rvecs, and tvecs
    print text_colors.OKGREEN + "Success." + text_colors.ENDCOLOR
    return (flag_success, position, orientation, rvecs, tvecs)
  else:
    return (flag_success, None, None, None, None)

class PnPSolver(object):
  def __init__(self):
    self._prev_sol_exists = False
    self._prev_rvecs = None
    self._prev_tvecs = None

    # Get camera matrix and set distortion coeficients to None since we rectified the image before finding centroids
    cam_info = rospy.wait_for_message("/camera/camera_info",CameraInfo)
    self._mtx = np.array([cam_info.K[0:3], cam_info.K[3:6], cam_info.K[6:9]])
    self._distortion = None

    # Publishers
    self._pub_pose = rospy.Publisher("/my_pos_data",PoseEstimate,queue_size=1)

    # Subscribers
    self._sub_filtered_centroids = rospy.Subscriber("/centroids/filtered",Centroids,self.cbFiltCentroids)

  def cbPrevSol(self,data):
    self._prev_sol_exists = data.found_solution
    self._prev_rvecs = data.rvecs
    self._prev_tvecs = data.tvecs

  def cbFiltCentroids(self,data):
    centroids = np.array(data.centroids)
    centroids = centroids.reshape(len(centroids)/2,2)

    self._prev_sol_exists, position, orientation, self._prev_rvecs,self._prev_tvecs = pose_extraction(centroids, self._prev_sol_exists, self._prev_rvecs, self._prev_tvecs, self._mtx, self._distortion)
    
    # Publish the position data
    pose = PoseEstimate()
    pose.orientation = orientation
    pose.position = position
    self._pub_pose.publish(pose)

    # Publish previous solution bounds

    raise Exception('NEED TO WRITE ORIENTATION TRANSFORM TO VICON SPACE AND ALSO SET PREVIOUS SOLUTION BOUNDS')

if __name__ == "__main__":
  # Initialize the node
  rospy.init_node('pnp_solver')

  # Create the node
  node = PnPSolver()

  # Spin
  rospy.spin()





































