#!/usr/bin/env python

# Set camera parameters in camera_params.yaml

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from itertools import combinations
import cProfile, pstats, StringIO
from creare.msg import FeaturePoints

#------------------------------------------------------------------------------#

# Helper Functions

#------------------------------------------------------------------------------#

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

def dist(coords1,coords2):
  '''
  Return the euclidean distance between two points with units as passed.
  '''
  a = coords1[0] - coords2[0]
  b = coords1[1] - coords2[1]
  distance = np.sqrt(a**2 + b**2)
  return distance

def cluster(data, maxgap):
  '''
  This method is taken from Raymond Hettinger at http://stackoverflow.com/questions/14783947/grouping-clustering-numbers-in-python

  Arrange data into groups where successive elements
  differ by no more than *maxgap*

  >>> cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
  [[1, 6, 9], [100, 102, 105, 109], [134, 139]]

  >>> cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)
  [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]
  '''

  data = np.array(data)
  data.sort()
  groups = [[data[0]]]
  for x in data[1:]:
    if np.absolute(x - groups[-1][-1]) <= maxgap:
      groups[-1].append(x)
    else:
      groups.append([x])
  return groups

def first_round_centroid_filter(centroids):
  '''
  This function takes in a list of centroids and returns a similar list that has been filtered to remove suspected erroneous values. Filtering is performed by clustering x and y image coordinates, and then intersecting those clusters. 
  '''

  def one_sub_cluster_of_at_least_8(clstr):
    ret_val = False
    for c in clstr:
      if len(c) >= 8:
        ret_val = True
    return ret_val

  # First, create two clusters of the centroids, one which is clustered by x and one by y. 
  x_coords = [c[0] for c in centroids]
  y_coords = [c[1] for c in centroids]

  std_dev_x = np.std(x_coords)
  std_dev_y = np.std(y_coords)

  k = 0.1
  k_step = 0.01
  x_clstr = cluster(x_coords,std_dev_x*k)
  y_clstr = cluster(y_coords,std_dev_y*k)

  x_iter = 1
  while not one_sub_cluster_of_at_least_8(x_clstr):
    x_clstr = cluster(x_coords,std_dev_x*k)
    k = k + k_step
    x_iter = x_iter + 1

  y_iter = 1
  while not one_sub_cluster_of_at_least_8(y_clstr):
    y_clstr = cluster(y_coords,std_dev_y*k)
    k = k + k_step
    y_iter = y_iter + 1

  # Since we specified that at least one of our clusters in both x and y have at least 8 points, we need to find the x and y cluster_of_interest that contains the 8 points. With a significant amount of noise, it is possible that we have more than one cluster with 8 points. Print a warning if this happens.
  x_cluster_of_interest = []
  y_cluster_of_interest = []
  for x in x_clstr:
    if len(x) >= 8:
      x_cluster_of_interest.append(x)
  for y in y_clstr:
    if len(y) >= 8:
      y_cluster_of_interest.append(y)

  if len(x_cluster_of_interest) > 1:
    print text_colors.WARNING + 'Warning: Too many x clusters of interest.' + text_colors.ENDCOLOR
    return []

  if len(y_cluster_of_interest) > 1:
    print text_colors.WARNING + 'Warning: Too many y clusters of interest.' + text_colors.ENDCOLOR
    return []

  x_cluster_of_interest = x_cluster_of_interest[0]
  y_cluster_of_interest = y_cluster_of_interest[0]

  # Gather centroids of interest from clusters of interest
  x_centroids_of_interest = set([])
  for x_val in x_cluster_of_interest:
    indices = [i for i,x in enumerate(x_coords) if x == x_val]
    for index in indices:
      y_val = y_coords[index]
      centroid_to_add = (x_val,y_val)
      x_centroids_of_interest.add(centroid_to_add)
  y_centroids_of_interest = set([])
  for y_val in y_cluster_of_interest:
    indices = [i for i,y in enumerate(y_coords) if y == y_val]
    for index in indices:
      x_val = x_coords[index]
      centroid_to_add = (x_val,y_val)
      y_centroids_of_interest.add(centroid_to_add)

  if self._show_images:
    img_xcoi = np.array(img)
    for c in x_centroids_of_interest:
      pt = (int(c[0]),int(c[1]))
      cv2.circle(img_xcoi, pt, 3, [0, 0, 255], 5)
    img_ycoi = np.array(img)
    for c in y_centroids_of_interest:
      pt = (int(c[0]),int(c[1]))
      cv2.circle(img_ycoi, pt, 3, [0, 0, 255], 5)
    img_to_show = np.hstack([np.asarray(img_xcoi), np.asarray(img_ycoi)])
    cv2.imshow('centroids of interest (x left, yright)',img_to_show)
    cv2.waitKey(1)

  # Now that we have clustered by x and y centroids, we need to take the intersection of these clusters, as this is likely our feature.
  centroids_of_interest = [c for c in y_centroids_of_interest if c in x_centroids_of_interest]

  # Attempt to recover grouping in the case where there are not 8 centroids shared by the x and y clusters.
  if len(centroids_of_interest) < 8 and len(centroids_of_interest) > 0:
    # Here we have the case where the number of centroids shared by x and y clusters is less than the number of feature points. This means that the x and y clusters are being thrown off by false positive measurements. At this point, we must decide which 8 points are the correct centroids.

    # First, we take the points that are shared by the x and y clusters. We calculate the average position of these points and use distance to this average position as a metric for choosing the remaining correct centroids.
    avg_pos_x = 0
    avg_pos_y = 0
    for c in centroids_of_interest:
      avg_pos_x = avg_pos_x + c[0]
      avg_pos_y = avg_pos_y + c[1]
    avg_pos_x = avg_pos_x/len(centroids_of_interest)
    avg_pos_y = avg_pos_y/len(centroids_of_interest)
    avg_pos = (avg_pos_x,avg_pos_y)

    dist_to_avg_pos = []
    for c in centroids_of_interest:
      dist_to_avg_pos.append(dist(c,avg_pos))

    dist_to_avg_pos_mean = np.mean(dist_to_avg_pos)
    dist_to_avg_pos_std = np.std(dist_to_avg_pos)

    # Now that we have the average position of the accepted centroids, we must query the remaining centroids for those that are nearest to this average position. 
    remaining_centroids = [c for c in centroids if c not in centroids_of_interest]

    selected_centroids = []
    for c in remaining_centroids:
      if np.absolute(dist(c,avg_pos) - dist_to_avg_pos_mean) < 5*dist_to_avg_pos_std:
        selected_centroids.append(c)

    for c in selected_centroids:
      centroids_of_interest.append(c)

  return centroids_of_interest

def second_round_centroid_filter(centroids):
  '''
  This function takes in a list of centroids and returns a similar list that has been filtered to remove suspected erroneous values. Filtering is performed by clustering according to pairwise slope value. 
  '''

  def at_least_one_cluster_of_at_least_12(clstr):
    for c in clstr:
      if len(c) >= 12:
        return True
    return False

  # Create a list of all subsets of 2 points
  subsets = combinations(centroids,2)

  # Calculate the slope of each pair
  slopes_and_points = []
  slopes = []
  for s in subsets:
    # Set point 1 to be the leftmost point and point 0 to be the right most point
    if s[0][0] < s[1][0]:
      pt0 = s[0]
      pt1 = s[1]
    else:
      pt0 = s[1]
      pt1 = s[0]
    # Determine the slope of the line. Handle special cases of slopes.
    rise = pt1[1]-pt0[1]
    run = pt1[0]-pt0[0]
    if run == 0 and rise == 0:
      # Do nothing. We are using the same point twice for some reason
      pass
    elif run == 0:
      # Do nothing. This is a vertical line and therefore is a pair of points with the points on different rows.
      pass
    else:
      # Store the slope and points together
      m = rise/run
      slopes_and_points.append((m,pt0,pt1))
      slopes.append(m)

  # Search the slopes_and_points list for point combinations which have nearly the same slope
  k = 0.005
  k_step = 0.005
  clustered_slopes = cluster(slopes,k)

  while not at_least_one_cluster_of_at_least_12(clustered_slopes) and k < 100*k_step:
    k = k + k_step
    clustered_slopes = cluster(slopes,k)

  slopes_of_interest = None
  for c in clustered_slopes:
    if len(c) >= 12:
      slopes_of_interest = c

  # Report an error if we do not detect a slope cluster with at least 12 points
  if slopes_of_interest is None:
    print text_colors.WARNING + 'Warning: Invalid slope clusters.' + text_colors.ENDCOLOR
    return []

  # Now that we have clustered by slope value, remove all subsets whose slope is not in the cluster
  for i in range(len(slopes_and_points)-1,-1,-1):
    if slopes_and_points[i][0] not in slopes_of_interest:
      del slopes_and_points[i]

  # Create a set of all of the points in our slope cluster so we have no duplicates.
  points = set([])
  for i in range(0,len(slopes_and_points)):
    pt0 = slopes_and_points[i][1]
    pt1 = slopes_and_points[i][2]
    points.add(pt0)
    points.add(pt1)

  return list(points)

def assign_points2(centroids):
  '''
  This function takes in a list of centroids and attempts to assign each of them to a corresponding real-world feature point. To do so, it finds the two subsets of four centroids with the lowest residual after a linear fit. The feature is two lines of four points, so there should be two subsets with relatively low residuals.
  '''

  subsets = combinations(centroids,4)
  rows = []
  k = 0
  # Find the first subset of 4 with low residual
  for s in subsets:
    if len(rows) < 1:
      x = [p[0] for p in s]
      y = [p[1] for p in s]
      _, residuals, _, _, _ = np.polyfit(x, y, 1, full = True)
      if residuals < 1:
        rows.append(list(s))
      k = k + 1

  # If we didn't find any subsets with a low enough residual, return empty assignments
  if len(rows) < 1:
    return [[0],[0]]

  # Now that we have assigned one row, we can deduce the other row. Subtract the chosen row from the list of centroids. The remaining 4 points are the other row. 
  other_row = [x for x in centroids if x not in rows[0]]
  rows.append(list(other_row))

  # Now we have both rows, so we must decide which is the top row and which is the bottom. First, sort each row so that the points in each row are organized from right to left in the image.
  for r in rows:
    r.sort(key=lambda x: x[0])

  # Use the first element of each row to determine which row is on top
  if rows[0][0][1] < rows[1][0][1]:
    top_row    = rows[0]
    bottom_row = rows[1]
  else:
    top_row    = rows[1]
    bottom_row = rows[0]

  if self._show_images:
    img_temp = np.array(img)
    for c in top_row:
      pt = (int(c[0]),int(c[1]))
      cv2.circle(img_temp, pt, 3, 255, 2)
    for c in bottom_row:
      pt = (int(c[0]),int(c[1]))
      cv2.circle(img_temp, pt, 3, 255, 1)
    cv2.imshow('row assignments (top is thicker)', img_temp)
    cv2.waitKey(1)
  
  return [bottom_row,top_row]

def feature_extraction(img_cv,prev_solution, mtx, distortion):

  img = img_cv
  
  if len(centroids) < 8:
    print text_colors.FAIL + "Failure: Too few centroids after initial selection." + text_colors.ENDCOLOR
    # cv2.waitKey(0)
    #return (img_ret, found_feature, position_estimate, feature_points, rvecs, tvecs)
    return (img,False,None,None,None,None)



  #----------------------------------------------------------------------------#

  # Filter the centroids to Obtain Only 8 Points of Interest

  #----------------------------------------------------------------------------#

  # print 'pre-filter',len(centroids)
  # img_temp = np.array(img_cv)
  # for c in centroids:
  #   pt = (int(c[0]),int(c[1]))
  #   cv2.circle(img_temp, pt, 3, 255, 1)
  # cv2.imshow('pre-filter',img_temp)
  # cv2.waitKey(1)

  # We expect 8 feature points, so we try to remove any extras
  if len(centroids) > 8:

    print text_colors.OKBLUE + "Note: First filter applied." + text_colors.ENDCOLOR
    centroids = first_round_centroid_filter(centroids)
    #centroids = first_round_centroid_filter(centroids,np.array(img_cv))

    # print 'after first round',len(centroids)
    # img_temp = np.array(img_cv)
    for c in centroids:
      pt = (int(c[0]),int(c[1]))
      cv2.circle(img_temp, pt, 3, [0, 0, 255], 5)
    # cv2.imshow('after first round',img_temp)
    # cv2.waitKey(1)

    if len(centroids) > 8:
      print text_colors.OKBLUE + "Note: Second filter applied." + text_colors.ENDCOLOR
      centroids = second_round_centroid_filter(centroids)

      # print 'after second round',len(centroids)
      # img_temp = np.array(img_cv)
      # for c in centroids:
      #   pt = (int(c[0]),int(c[1]))
      #   cv2.circle(img_temp, pt, 3, [0, 0, 255], 5)
      # cv2.imshow('after second round',img_temp)
      # cv2.waitKey(1)

  # print ''

  # print len(centroids)

  # img_temp = np.array(img_cv)
  # for c in centroids:
  #   pt = (int(c[0]),int(c[1]))
  #   cv2.circle(img_temp, pt, 3, 255, 1)
  # cv2.imshow('post-filter',img_temp)
  # cv2.waitKey(1)

  # img_to_show2 = img_temp

  # img_to_show = np.hstack([np.asarray(img_to_show1), np.asarray(img_to_show2)])
  # cv2.imshow('Before (Left) and After (Right) Noise Processing',img_to_show)
  # cv2.waitKey(1)

  if len(centroids) < 8:
    print text_colors.FAIL + "Failure: Too few centroids after filtering." + text_colors.ENDCOLOR
    #return (img_ret, found_feature, position_estimate, feature_points, rvecs, tvecs)
    return (img,False,None,None,None,None)



  #----------------------------------------------------------------------------#

  # Finding the Feature

  #----------------------------------------------------------------------------#

  # This section assigns the centroids that we found to the feature points.

  assigned_points = assign_points2(centroids)#,np.array(img_cv))

  total_points_assigned = len(assigned_points[0]) + len(assigned_points[1])
  if not total_points_assigned == 8:
    print text_colors.FAIL + "Failure: Failure to assign points correctly." + text_colors.ENDCOLOR

    # print len(assigned_points),len(assigned_points[0]),len(assigned_points[1])

    # img_temp2 = np.array(img_cv)
    # thicknesses = [1,2]
    # k = 0
    # for list_of_points in assigned_points:
    #   for c in list_of_points:
    #     img_temp = np.array(img_cv)
    #     pt = (int(c[0]),int(c[1]))
    #     cv2.circle(img_temp, pt, 3, 255, thicknesses[k])
    #     cv2.circle(img_temp2, pt, 3, 255, thicknesses[k])
    #     cv2.imshow('not 8 points to assign',img_temp)
    #     cv2.waitKey(0)
    #   k = k + 1
    # cv2.imshow('not 8 points to assign',img_temp2)
    # cv2.waitKey(0)

    #return (img_ret, found_feature, position_estimate, feature_points, rvecs, tvecs)
    return (img,False,None,None,None,None)

  # # Show the assigned points on the image. Bottom row then top row, always left to right.
  # img_temp = np.array(img_cv)
  # thicknesses = [1,2]
  # k = 0
  # for row in assigned_points:
  #   for p in row:
  #     pt = (int(p[0]),int(p[1]))
  #     cv2.circle(img_temp, pt, 3, 255, thicknesses[k])
  #     cv2.imshow('bottom_row_then_top_left_to_right',img_temp)
  #     cv2.waitKey(0)
  #   k = k+1

  # With a full point correspondence, we can continue to the PnP solver and find 
  # the distance to our feature.



  #----------------------------------------------------------------------------#

  # Calculate Distance

  #----------------------------------------------------------------------------#

  # Define object points. The units used here, [cm], will determine the units of the output.
  objp = np.zeros((8,1,3), np.float32)

  # Briggs Field Testing [m]
  width = 6 # x-direction
  length = 30 # y-direction
  height = 2.4 # z-direction
  # 1:35 Testing [m]
  width = 6 # x-direction
  length = 30 # y-direction
  height = 2.4 # z-direction
  # Lowermost Subset
  objp[0] = [ 0*width, 0*length, 0*height]
  objp[1] = [ 1*width, 0*length, 0*height]
  objp[2] = [ 2*width, 0*length, 0*height]
  objp[3] = [ 3*width, 0*length, 0*height]
  # Uppermost Subset
  objp[4] = [ 0*width, 1*length, 1*height]
  objp[5] = [ 1*width, 1*length, 1*height]
  objp[6] = [ 2*width, 1*length, 1*height]
  objp[7] = [ 3*width, 1*length, 1*height]

  # Define feature points by the correspondences determined above
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
  axis = np.float32([[10,0,0], [0,10,0], [0,0,10]]).reshape(-1,3)

  # Find rotation and translation vectors
  prev_rvecs = None
  prev_tvecs = None
  use_prev_solution = False
  if prev_sol_exists:
    prev_rvecs = np.array(prev_solution[-3])
    prev_tvecs = np.array(prev_solution[-2])
    use_prev_solution = True
  ret,rvecs,tvecs = cv2.solvePnP(objp,feature_points,mtx,distortion,prev_rvecs,prev_tvecs,use_prev_solution)

  # rvecs,tvecs,inliers = cv2.solvePnPRansac(objp,feature_points,mtx,distortion)
  # _,rvecs,tvecs,inliers = cv2.solvePnPRansac(objp,feature_points,mtx,distortion)
  
  # # Project 3D points onto image plane
  # imgpts,jac = cv2.projectPoints(axis,rvecs,tvecs,mtx,distortion)
  # img_axes = draw(img,feature_points,imgpts)
  # img_axes = img
  
  # # Undistortion and cropping
  # h,w = img_axes.shape[:2]
  # mtx_new,roi = cv2.getOptimalNewCameraMatrix(mtx,distortion,(w,h),1,(w,h))
  # img_axes = cv2.undistort(img_axes,mtx,distortion,None,mtx_new)
  # x,y,w,h = roi
  # img_axes = img_axes[y:y+h, x:x+w]
  
  # Calculate position
  Pc = tuple(feature_points[0].ravel())
  Pc = np.array([[Pc[0]], [Pc[1]], [1]])
  Kinv = np.matrix(np.linalg.inv(mtx))
  R,_ = cv2.Rodrigues(rvecs)
  Rinv = np.matrix(np.linalg.inv(R))
  T = np.array(tvecs)
  Pw = Rinv*(Kinv*Pc-T)

  if profiler_on:
    # Stop the profiler and print the results
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

  # # Show image for debugging if desired
  # cv2.imshow('axes',img_axes)
  # cv2.waitKey(1)

  # Return the obtained x,y,z points, the image with the axes drawn on it, a flag stating that
  # we found the image successfully, and the feature_points array.
  print text_colors.OKGREEN + "Success." + text_colors.ENDCOLOR
  return (img, True, Pw, feature_points, rvecs, tvecs)

class ImageProcessor(object):
  def __init__(self):
    self.found_one_solution = False

    # Previous Solution
    self.prev_solution = [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1,-1],[-1,-1,-1],False]

    # Image converter
    self.bridge = CvBridge()

    # Publishers
    self.pub_prev_sol = rospy.Publisher("/prev_solution",FeaturePoints,queue_size=1)

    # Subscribers
    self.sub_img_raw  = rospy.Subscriber("/camera/image_raw",Image,self.cbImgRect)
    self.sub_prev_sol  = rospy.Subscriber("/prev_solution",FeaturePoints,self.cbPrevSol)

  def cbPrevSol(self,data):
    # Lowermost Subset
    self.prev_solution[0] = [data.bottom_left_x,data.bottom_left_y]
    self.prev_solution[1] = [data.bottom_middle_left_x,data.bottom_middle_left_y]
    self.prev_solution[2] = [data.bottom_middle_right_x,data.bottom_middle_right_y]
    self.prev_solution[3] = [data.bottom_right_x,data.bottom_right_y]
    # Uppermost Subset
    self.prev_solution[4] = [data.top_left_x,data.top_left_y]
    self.prev_solution[5] = [data.top_middle_left_x,data.top_middle_left_y]
    self.prev_solution[6] = [data.top_middle_right_x,data.top_middle_right_y]
    self.prev_solution[7] = [data.top_right_x,data.top_right_y]
    # Rvecs
    self.prev_solution[8] = [[data.rvecs1],[data.rvecs2],[data.rvecs3]]
    # Tvecs
    self.prev_solution[9] = [[data.tvecs1],[data.tvecs2],[data.tvecs3]]
    # Found Feature
    self.prev_solution[10] = data.found_solution

  def cbImgRect(self,data):
    # Collect time of when image was received
    time = rospy.get_rostime()

    # Convert the ROS image to an OpenCV image
    try:
      img = self.bridge.imgmsg_to_cv2(data, '8UC1')
    except CvBridgeError as e:
      print text_colors.WARNING + 'Warning: Image converted unsuccessfully before processing.' + text_colors.ENDCOLOR

    # Remove the edge of the image since there are erroneous pixels in the top left of the image (some sort of image conversion error perhaps?)
    edge_buffer = 1
    img = img[edge_buffer:img.shape[0] - edge_buffer, edge_buffer:img.shape[1] - edge_buffer]

    # Find feature
    found_feature, position_estimate, feature_points, rvecs, tvecs = feature_extraction(img,self.prev_solution,self.mtx,self.distortion)

    # # Augment the image to show the position estimate
    # if False:#True
    #   if found_feature:
    #     x = position_estimate[0]
    #     y = position_estimate[1]
    #     z = position_estimate[2]
    #     distance_estimate = np.sqrt(x**2 + y**2 + z**2)

    #     x_str = 'x:'
    #     y_str = 'y:'
    #     z_str = 'z:'
    #     dist_str = 'd:'

    #     x_num = '%7.2f m' % x
    #     y_num = '%7.2f m' % y
    #     z_num = '%7.2f m' % z
    #     dist_num = '%7.2f m' % distance_estimate

    #     cv2.putText(img_ret,x_str,(10,25),font,1,255,1,cv2.CV_AA)
    #     cv2.putText(img_ret,y_str,(10,50),font,1,255,1,cv2.CV_AA)
    #     cv2.putText(img_ret,z_str,(10,75),font,1,255,1,cv2.CV_AA)
    #     cv2.putText(img_ret,dist_str,(10,100),font,1,255,1,cv2.CV_AA)

    #     cv2.putText(img_ret,x_num,(35,25),font,1,255,1,cv2.CV_AA)
    #     cv2.putText(img_ret,y_num,(35,50),font,1,255,1,cv2.CV_AA)
    #     cv2.putText(img_ret,z_num,(35,75),font,1,255,1,cv2.CV_AA)
    #     cv2.putText(img_ret,dist_num,(35,100),font,1,255,1,cv2.CV_AA)
    #   else:
    #     cv2.putText(img_ret,' Feature',(27,50),font,1,255,1,cv2.CV_AA)
    #     cv2.putText(img_ret,'Not Found',(25,75),font,1,255,1,cv2.CV_AA)
    #     pass

    # prev_sol = FeaturePoints()

    # if found_feature:
    #   # Calculate distance
    #   x = position_estimate[0]
    #   y = position_estimate[1]
    #   z = position_estimate[2]
    #   distance_estimate = np.sqrt(x**2 + y**2 + z**2)

    #   # Set the solution for the next loop
    #   prev_sol.found_solution = True

    #   prev_sol.bottom_left_x         = feature_points[0][0][0]
    #   prev_sol.bottom_left_y         = feature_points[0][0][1]
    #   prev_sol.bottom_middle_left_x  = feature_points[1][0][0]
    #   prev_sol.bottom_middle_left_y  = feature_points[1][0][1]
    #   prev_sol.bottom_middle_right_x = feature_points[2][0][0]
    #   prev_sol.bottom_middle_right_y = feature_points[2][0][1]
    #   prev_sol.bottom_right_x        = feature_points[3][0][0]
    #   prev_sol.bottom_right_y        = feature_points[3][0][1]
    #   prev_sol.top_left_x            = feature_points[4][0][0]
    #   prev_sol.top_left_y            = feature_points[4][0][1]
    #   prev_sol.top_middle_left_x     = feature_points[5][0][0]
    #   prev_sol.top_middle_left_y     = feature_points[5][0][1]
    #   prev_sol.top_middle_right_x    = feature_points[6][0][0]
    #   prev_sol.top_middle_right_y    = feature_points[6][0][1]
    #   prev_sol.top_right_x           = feature_points[7][0][0]
    #   prev_sol.top_right_y           = feature_points[7][0][1]

    #   prev_sol.rvecs1 = rvecs[0]
    #   prev_sol.rvecs2 = rvecs[1]
    #   prev_sol.rvecs3 = rvecs[2]

    #   prev_sol.tvecs1 = tvecs[0]
    #   prev_sol.tvecs2 = tvecs[1]
    #   prev_sol.tvecs3 = tvecs[2]

    #   self.found_one_solution = True

    #   # Create a string for publishing
    #   # data_str = ('[SUCCESS]\n\ttime:\n\t\tsecs: %s\n\t\tnsecs: %s\n' %(time.secs,time.nsecs))
    #   data_str = ('[SUCCESS]\n\ttime:\n\t\tsecs: %s\n\t\tnsecs: %s\n\tposition' \
    #       '\n\t\tx: %s\n\t\ty: %s\n\t\tz: %s\n' %(time.secs,time.nsecs,x,y,z))
    # elif self.found_one_solution:
    #   # Set the solution for the next loop

    #   # Set this to true even though we did not find a solution on this particular loop. This should be 
    #   # rewritten to be more clear, but essentially, if we did not find a solution on this loop then we 
    #   # should use the most recent solution that we did find. To use that solution, we set this to true. 
    #   prev_sol.found_solution = True

    #   prev_sol.bottom_left_x         = self.prev_solution[0][0]
    #   prev_sol.bottom_left_y         = self.prev_solution[0][1]
    #   prev_sol.bottom_middle_left_x  = self.prev_solution[1][0]
    #   prev_sol.bottom_middle_left_y  = self.prev_solution[1][1]
    #   prev_sol.bottom_middle_right_x = self.prev_solution[2][0]
    #   prev_sol.bottom_middle_right_y = self.prev_solution[2][1]
    #   prev_sol.bottom_right_x        = self.prev_solution[3][0]
    #   prev_sol.bottom_right_y        = self.prev_solution[3][1]
    #   prev_sol.top_left_x            = self.prev_solution[4][0]
    #   prev_sol.top_left_y            = self.prev_solution[4][1]
    #   prev_sol.top_middle_left_x     = self.prev_solution[5][0]
    #   prev_sol.top_middle_left_y     = self.prev_solution[5][1]
    #   prev_sol.top_middle_right_x    = self.prev_solution[6][0]
    #   prev_sol.top_middle_right_y    = self.prev_solution[6][1]
    #   prev_sol.top_right_x           = self.prev_solution[7][0]
    #   prev_sol.top_right_y           = self.prev_solution[7][1]

    #   prev_sol.rvecs1 = self.prev_solution[8][0][0]
    #   prev_sol.rvecs2 = self.prev_solution[8][1][0]
    #   prev_sol.rvecs3 = self.prev_solution[8][2][0]

    #   prev_sol.tvecs1 = self.prev_solution[9][0][0]
    #   prev_sol.tvecs2 = self.prev_solution[9][1][0]
    #   prev_sol.tvecs3 = self.prev_solution[9][2][0]

    #   # Create a string for publishing
    #   data_str = ('[FAILURE]\n\ttime:\n\t\tsecs: %s\n\t\tnsecs: %s\n' %(time.secs,time.nsecs))
    # else:
    #   # Create a string for publishing
    #   data_str = ('[FAILURE]\n\ttime:\n\t\tsecs: %s\n\t\tnsecs: %s\n' %(time.secs,time.nsecs))
        
    # want_to_publish_image = False
    # if want_to_publish_image:
    #   img_ret = cv2.cvtColor(img_ret, cv2.COLOR_GRAY2BGR)

    #   # Convert the OpenCV image to a ROS image
    #   converted_successfully = True
    #   try:
    #     data = self.bridge.cv2_to_imgmsg(img_ret, 'bgr8')
    #   except:
    #     converted_successfully = False
    #     print text_colors.WARNING + 'Warning: Image converted unsuccessfully after processing.' + text_colors.ENDCOLOR
    #     # rospy.loginfo(e)

    #   # Publish the image with any modifications made (drawn axes, etc.)
    #   if converted_successfully:
    #     self.pub_img_proc.publish(data)

    # # Publish the previous solution
    # self.pub_prev_sol.publish(prev_sol)

    # # Publish the position data as a string
    # self.pub_pos_as_str.publish(data_str)

    # if outer_debug_flag:
    #   pr.disable()
    #   s = StringIO.StringIO()
    #   sortby = 'cumulative'
    #   ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #   ps.print_stats()
    #   print s.getvalue()

if __name__ == "__main__":
  # Initialize the node
  rospy.init_node('feature_extractor')

  # Create the node
  node = ImageProcessor()

  # Spin
  rospy.spin()
