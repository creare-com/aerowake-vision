#!/usr/bin/env python

import rospy
import numpy as np
from itertools import combinations
from creare.msg import Centroids

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

  return np.array(centroids_of_interest)

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

  return np.array(list(points))

def noise_filtering(centroids):
  '''
  This function takes in an array of centroids and filters it for noise. The feature is expected to have eight points arranged in two parallel rows of 4 equally spaced points.
  '''

  if len(centroids) < 8:
    print text_colors.FAIL + "Failure: Too few centroids after initial selection." + text_colors.ENDCOLOR
    return np.array([])

  # We expect 8 feature points, so we try to remove any extras
  if len(centroids) > 8:
    print text_colors.OKBLUE + "Note: First filter applied." + text_colors.ENDCOLOR
    centroids = first_round_centroid_filter(centroids)

    if len(centroids) > 8:
      print text_colors.OKBLUE + "Note: Second filter applied." + text_colors.ENDCOLOR
      centroids = second_round_centroid_filter(centroids)

  # The filters may have wiped out too many points. Check if that's the case.
  if len(centroids) < 8:
    print text_colors.FAIL + "Failure: Too few centroids after filtering." + text_colors.ENDCOLOR
    return np.array([])

  return centroids

class NoiseFilter(object):
  def __init__(self):
    # Publishers
    self._pub_filtered_centroids = rospy.Publisher("/centroids/filtered",Centroids,queue_size=1)

    # Subscribers
    self._sub_initial_centroids = rospy.Subscriber("/centroids/initial",Centroids,self.cbInitCentroids)

  def cbInitCentroids(self,data):
    '''
    This callback filters the list of centroids for noise and publishes the result.
    '''

    centroids = np.array(data.centroids)
    centroids = centroids.reshape(len(centroids)/2,2)

    if len(data.centroids) > 0:
      centroids_filtered = noise_filtering(centroids)
    else:
      centroids_filtered = []

    if len(centroids_filtered) > 0:
      num_rows = len(centroids_filtered)
      num_cols = len(centroids_filtered[0])
      centroids_filtered = centroids_filtered.reshape(1,num_rows*num_cols).tolist()[0]

    self._pub_filtered_centroids.publish(centroids_filtered)

if __name__ == "__main__":
  # Initialize the node
  rospy.init_node('noise_filter')

  # Create the node
  node = NoiseFilter()

  # Spin
  rospy.spin()
