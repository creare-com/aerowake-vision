#!/usr/bin/env python

# USE camera_params.yaml TO SET THE FOLLOWING VALUES:
#   VIDEO_MODE: Format7_Mode0 (format7_mode0)
#   FRAMERATE: 30 (MAX)
#   EXPOSURE: -7.58496 (MIN)
#   SHUTTER: 0.01
#   GAIN: -2.25
#   WHITE_BALANCE_BLUE: 800
#   WHITE_BALANCE_RED: 450

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
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
    Return the euclidean distance between two 2D points.
    '''
    a = coords1[0] - coords2[0]
    b = coords1[1] - coords2[1]
    distance = np.sqrt(a**2 + b**2)
    return distance

def draw(img,corners,imgpts):
    img_deep_copy = np.array(img)
    corner = tuple(corners[0].ravel())

    cv2.line(img_deep_copy,corner,tuple(imgpts[0].ravel()), \
        (255,0,0),5)
    cv2.line(img_deep_copy,corner,tuple(imgpts[1].ravel()), \
        (0,255,0),5)
    cv2.line(img_deep_copy,corner,tuple(imgpts[2].ravel()), \
        (0,0,255),5)
    return img_deep_copy

def cluster(data, maxgap):
    '''
    This method is taken from Raymond Hettinger at http://stackoverflow.com/questions/14783947/grouping-clustering-numbers-in-python
    '''
    '''Arrange data into groups where successive elements
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

def obtain_initial_centroids(img,prev_sol = None):
    '''
    This function takes in an image and returns an array of the positions of any contours that show up after filtering for green. 
    '''

    # Initialize the region of interest. The region_of_interest[0] and region_of_interest[2] elements will be added to the location of each centroid, 
    # so this needs to be initialized to zero for the case when there does not exist a previous solution. 
    region_of_interest = [0, 0, 0, 0]

    # If the previous solution exists, use it to crop the image
    if not prev_sol is None:
        # Create a block around each of the points in the previous solution. These blocks are the only parts that we will
        # blur and then search for the new point.
        prev_points = prev_sol[:-3]

        # Define the boundaries of the region of interest
        left_most = np.minimum(prev_points[0][0],prev_points[4][0])
        right_most = np.maximum(prev_points[3][0],prev_points[7][0])
        upper_most = np.minimum(prev_points[4][1],prev_points[7][1])
        lower_most = np.maximum(prev_points[0][1],prev_points[3][1])
        border = 150
        region_of_interest = [left_most - border, right_most + border, upper_most - border, lower_most + border]
        for i in range(0,len(region_of_interest)):
            region_of_interest[i] = int(region_of_interest[i])

        # Ensure none of the boundaries are outside of the image
        if region_of_interest[0] < 0:
            region_of_interest[0] = 0
        if region_of_interest[1] >= img.shape[1]:
            region_of_interest[1] = img.shape[1] - 1
        if region_of_interest[2] < 0:
            region_of_interest[2] = 0
        if region_of_interest[3] >= img.shape[0]:
            region_of_interest[3] = img.shape[0] - 1

        # Cut the image to only be the region of interest
        img = img[region_of_interest[2]:region_of_interest[3],region_of_interest[0]:region_of_interest[1],:]

    # Blur the image
    kernelX = np.array([0.3,0.4,0.3])
    kernelY = np.array([0.3,0.4,0.3])
    img = cv2.sepFilter2D(img,-1,kernelX,kernelY)

    # cv2.imshow('blur',img)
    # cv2.waitKey(1)

    # Convert the image to HSV
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # Define the boundaries for what is considered green (HSV format)
    # 144-191   70-79   26-67
    # 135-159   90-94   24-53
    # 131-201   83-95   17-44
    # 
    # 131-201   70-95   17-67
    # 65-101    178-242 43-165*
    green_limit_lower0 = np.array([65,178,43])
    green_limit_upper0 = np.array([101,242,165])



    # # Define the boundaries for 100W Beacon with gain 0, shutter min

    # # Far (25 m)
    # # 141-171    71-82    49-73
    # # 70-86      181-210  124-187

    # green_limit_lower0 = np.array([69,179,122])
    # green_limit_upper0 = np.array([88,212,189])



    # Find green within the image
    mask = cv2.inRange(img, green_limit_lower0, green_limit_upper0)
    img = cv2.bitwise_and(img, img, mask = mask)

    # cv2.imshow('green',img)
    # cv2.waitKey(1)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('gray',img)
    # cv2.waitKey(1)

    # Convert to binary
    max_value = 255
    block_size = 5
    const = 1
    threshold_value = 30
    _,img = cv2.threshold(img,threshold_value,max_value,cv2.THRESH_BINARY)

    # cv2.imshow('binary',img)
    # cv2.waitKey(1)

    # Morph image to 'close' the shapes that are found
    kernel = np.ones((2,2),np.uint8)
    img = cv2.dilate(img,kernel,iterations = 2)
    img = cv2.erode(img,kernel,iterations = 2)

    # cv2.imshow('morph',img)
    # cv2.waitKey(1)

    # Find contours
    contours, _ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #_, contours, _ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    # Extract centroids
    centroids = []
    centroids_dist = []
    centroids_and_dist = []
    all_centers = set([])
    max_y,max_x = img.shape[:2]
    edge_of_img_border = 5
    nearby_restriction_value = 5
    for single_contour in contours:
        # For each contour, ensure that the area is greater than 1 pixel
        area = cv2.contourArea(single_contour)
        if area > 0.01:
            # Obtain the coordinates of the center of the contour
            (x,y),_ = cv2.minEnclosingCircle(single_contour)
            center = (int(x),int(y))
            # If an overhead light is right on the edge of the image, then it will occasionally produce a false positive reading.
            # Ignore points right on the edge of the image to get rid of this source of noise. Overhead lights that are elsewhere in 
            # the image are taken care of by the blurring of the image.
            on_edge_of_image = False
            if center[0] < edge_of_img_border or center[1] < edge_of_img_border or center[0] > max_x-edge_of_img_border or center[1] > max_y-edge_of_img_border:
                on_edge_of_image = True
            if not on_edge_of_image:
                # If we have already detected a point at a given coordinate, do not add it again. Since the center is cast to an integer, this
                # step ensures that redundant points are not counted. e.g. (617.5,742.2) and (617.9,742.4) may be detected as separate points, but clearly
                # they refer to the same point in the image and only one should be counted. 
                if center not in all_centers:
                    all_centers.add(center)
                    # Add nearby-restriction so that we do not get redundant centroids
                    for i in range(-nearby_restriction_value,nearby_restriction_value + 1):
                        for j in range(-nearby_restriction_value,nearby_restriction_value + 1):
                            nearby_center = (center[0] + i,center[1] + j)
                            all_centers.add(nearby_center)
                    # cv2.circle(img_temp, center, 3, [0, 0, 255], 5)
                    # cv2.imshow('obt init cntds',img_temp)
                    # cv2.waitKey(0)
                    centroids.append((x + region_of_interest[0],y + region_of_interest[2]))

    # Return the centroids of the points found
    return centroids

# def first_round_centroid_filter(centroids):
def first_round_centroid_filter(centroids,img):
    '''
    This function takes in an array of centroids of green areas and returns a similar array that has been filtered to remove erroneous values.
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

    # Since we specified that at least one of our clusters have at least 8 points, we need to find the x and y cluster_of_interest 
    # that contains the 8 points. With a significant amount of noise, it is possible that we have more than one cluster with 8 points.
    # Print an error if this happens.
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

    # Now that we have clustered by x and y coordinate, we need to take the intersection of these clusters, as this is likely our feature.
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

    # print 'x cntds of intrst',len(x_centroids_of_interest)
    # img_temp = np.array(img)
    # for c in x_centroids_of_interest:
    #     pt = (int(c[0]),int(c[1]))
    #     cv2.circle(img_temp, pt, 3, [0, 0, 255], 5)
    # cv2.imshow('x c o i',img_temp)
    # print 'y cntds of intrst',len(y_centroids_of_interest)
    # img_temp = np.array(img)
    # for c in y_centroids_of_interest:
    #     pt = (int(c[0]),int(c[1]))
    #     cv2.circle(img_temp, pt, 3, [0, 0, 255], 5)
    # cv2.imshow('y c o i',img_temp)
    # cv2.waitKey(1)

    centroids_of_interest = [c for c in y_centroids_of_interest if c in x_centroids_of_interest]

    if len(centroids_of_interest) < 8 and len(centroids_of_interest) > 0:
        # Here we have the case where the number of centroids shared by x and y clusters is less than the number
        # of feature points. This means that the x and y clusters are being thrown off by false positive measurements.
        # At this point, we must decide which 8 points are to be identified as the correct centroids.

        # First, we take the points that are shared by the x and y clusters. We calculate the average position of these
        # points and use distance to this average position as a metric for choosing the remaining correct centroids.
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

        # Now that we have the average position of the accepted centroids, we must query the remaining centroids
        # for those that are nearest to this average position. 
        remaining_centroids = [c for c in centroids if c not in centroids_of_interest]

        selected_centroids = []
        for c in remaining_centroids:
            if np.absolute(dist(c,avg_pos) - dist_to_avg_pos_mean) < 5*dist_to_avg_pos_std:
                selected_centroids.append(c)

        # img_temp = np.array(img)
        # for c in selected_centroids:
        #     pt = (int(c[0]),int(c[1]))
        #     cv2.circle(img_temp, pt, 3, [0, 0, 255], 5)
        # cv2.imshow('selected centroids',img_temp)
        # cv2.waitKey(1)

        for c in selected_centroids:
            centroids_of_interest.append(c)

    return centroids_of_interest

def second_round_centroid_filter(centroids):
    '''
    This function 
    '''

    # return centroids
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

    # Report an error if we do not detect a slope cluster with only 12 points
    if slopes_of_interest is None:
        print text_colors.WARNING + 'ERROR: Invalid slope clusters.' + text_colors.ENDCOLOR
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

# def assign_points(centroids):
def assign_points(centroids,img):

    def two_of_the_clusters_have_6_intercepts(clstr):
        k = 0

        for c in clstr:
            if len(c) == 6:
                k = k + 1

        return k == 2

    # Here we assume that we have only 8, correctly-selected centroids. As such, we have 2 rows of 4 points.
    # We start by clustering these points by their rows.

    subsets = combinations(centroids,2)

    # Set the reference point to just above and to the left of the points of interest. If we were to 
    # use the edge of the image to calculate the intercept, it may magnify the errors.
    min_x = np.min([c[0] for c in centroids])
    min_y = np.min([c[0] for c in centroids])

    buffer_val = 20
    reference_point = (min_x - buffer_val, min_y - buffer_val)

    # With respect to the reference point, calculate the y-intercept of each pair of points
    intercepts_and_points = []
    intercepts = []
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
            # Do nothing. We are using the same point twice for some reason.
            pass
        elif run == 0:
            # Do nothing. This is a vertical line and therefore is a pair of points with the points on different rows.
            pass
        else:
            m = rise/run

            # Use an average of the two points to minimize errors from any one point
            pt_avg = ((pt0[0]+pt1[0])/2 - reference_point[0],(pt0[1]+pt1[1])/2 - reference_point[1])
            # Calculate the y-intercept with the redefined point
            x = pt_avg[0]
            y = pt_avg[1]
            b = y - m*x
            intercepts_and_points.append((b,pt0,pt1))
            intercepts.append(b)

    # Cluster the intercepts, then separate intercepts and points based upon this clustering.
    k = 1
    k_step = 1
    k_max = 500*k_step
    clustered_intercepts = cluster(intercepts,k)

    while not two_of_the_clusters_have_6_intercepts(clustered_intercepts) and k <= k_max:
        k = k + k_step
        clustered_intercepts = cluster(intercepts,k)

    # print '\nk loop:',k
    # for c in clustered_intercepts:
    #     for pt in c:
    #         print pt
    #     print ''
    # print k, k_max
    # cv2.waitKey(1)

    if k > k_max:
        print text_colors.WARNING + 'Warning: Maximum iterations reached in intercept clustering.' + text_colors.ENDCOLOR
        # cv2.waitKey(0)
        return []

    # Determine the points that correspond the the clustered intercepts
    points_sorted_by_row = []
    tol = 0.00001
    for c in clustered_intercepts:
        if len(c) == 6:
            points_sorted_by_row.append(set([]))
            # img_temp = np.array(img)
            # origin_coords = (0,0)
            # cv2.circle(img_temp, origin_coords, 3, [255, 255, 255], 5)
            for b in c:
                index = [i for i, j in enumerate(intercepts) if np.absolute(j - b) < tol]
                index = index[0]
                pt0 = intercepts_and_points[index][1]
                pt1 = intercepts_and_points[index][2]
                points_sorted_by_row[-1].add(pt0)
                points_sorted_by_row[-1].add(pt1)
            #     pt00 = (int(pt0[0]),int(pt0[1]))
            #     pt11 = (int(pt1[0]),int(pt1[1]))
            #     cv2.circle(img_temp, pt00, 3, [0, 255, 0], 5)
            #     cv2.circle(img_temp, pt11, 3, [0, 0, 255], 5)
            # cv2.imshow('intercept test',img_temp)
            # cv2.waitKey(0)
            points_sorted_by_row[-1] = list(points_sorted_by_row[-1])

    # # Show points to screen
    # for pt_list in points_sorted_by_row:
    #     img_temp = np.array(img)
    #     for pt in pt_list:
    #         pt0 = (int(pt[0]),int(pt[1]))
    #         cv2.circle(img_temp, pt0, 3, [0, 0, 255], 5)
    #     cv2.imshow('intercept test',img_temp)
    #     cv2.waitKey(0)

    # Here we have our 8 centroids sorted into two rows of 4, so we can assign them based upon their location in the image

    for i in range(0,len(points_sorted_by_row)):
        points_sorted_by_row[i] = sorted(points_sorted_by_row[i],key = lambda x: x[1])

    # Now each row is sorted to have the uppermost point in the row as the 0th element.
    if points_sorted_by_row[0][0][1] < points_sorted_by_row[1][0][1]:
        top_row = points_sorted_by_row[0]
        bottom_row = points_sorted_by_row[1]
    else:
        top_row = points_sorted_by_row[1]
        bottom_row = points_sorted_by_row[0]

    # img_temp = np.array(img)
    # for c in top_row:
    #     pt = (int(c[0]),int(c[1]))
    #     cv2.circle(img_temp, pt, 3, [0, 0, 255], 5)
    # cv2.imshow('top_row',img_temp)
    # cv2.waitKey(1)

    # img_temp = np.array(img)
    # for c in bottom_row:
    #     pt = (int(c[0]),int(c[1]))
    #     cv2.circle(img_temp, pt, 3, [0, 0, 255], 5)
    # cv2.imshow('bottom_row',img_temp)
    # cv2.waitKey(1)

    top_row = sorted(top_row,key = lambda x: x[0])
    bottom_row = sorted(bottom_row,key = lambda x: x[0])

    return (bottom_row,top_row)

def feature_extraction(img_cv,prev_solution):
    debug_flag = False
    profiler_on = False

    if profiler_on:
        # Start the profiler (if desired)
        pr = cProfile.Profile()
        pr.enable()

    img = img_cv

    # img_temp = np.array(img_cv)
    # cv2.imshow('original',img_temp)
    # cv2.waitKey(1)



    #------------------------------------------------------------------------------#

    # Filter the Image to Obtain centroids of Green Contours

    #------------------------------------------------------------------------------#
    prev_sol_exists = prev_solution[-1]

    if prev_sol_exists:
        centroids = obtain_initial_centroids(img,prev_solution)
        if len(centroids) < 8:
            centroids = obtain_initial_centroids(img)
    else:
        centroids = obtain_initial_centroids(img)

    # print 'initial',len(centroids)
    img_temp = np.array(img_cv)
    for c in centroids:
        pt = (int(c[0]),int(c[1]))
        cv2.circle(img_temp, pt, 3, [0, 0, 255], 5)
    # cv2.imshow('initial',img_temp)
    # cv2.waitKey(1)

    img_to_show1 = img_temp
    
    if len(centroids) < 8:
        #return (img_ret, found_feature, position_estimate, feature_points, rvecs, tvecs)
        return (img,False,None,None,None,None)



    #------------------------------------------------------------------------------#

    # Filter the centroids to Obtain Only 8 Points of Interest

    #------------------------------------------------------------------------------#

    # # print 'pre-filter',len(centroids)
    # img_temp = np.array(img_cv)
    # for c in centroids:
    #     pt = (int(c[0]),int(c[1]))
    #     cv2.circle(img_temp, pt, 3, [0, 0, 255], 5)
    # cv2.imshow('pre-filter',img_temp)
    # cv2.waitKey(1)

    # We expect 8 feature points, so we try to remove any extras
    if len(centroids) > 8:
        # centroids = first_round_centroid_filter(centroids)
        centroids = first_round_centroid_filter(centroids,np.array(img_cv))
    
        # print 'after first round',len(centroids)
        # img_temp = np.array(img_cv)
        # for c in centroids:
        #     pt = (int(c[0]),int(c[1]))
        #     cv2.circle(img_temp, pt, 3, [0, 0, 255], 5)
        # cv2.imshow('after first round',img_temp)
        # cv2.waitKey(1)

        if len(centroids) > 8:
            centroids = second_round_centroid_filter(centroids)
    
            # print 'after second round',len(centroids)
            # img_temp = np.array(img_cv)
            # for c in centroids:
            #     pt = (int(c[0]),int(c[1]))
            #     cv2.circle(img_temp, pt, 3, [0, 0, 255], 5)
            # cv2.imshow('after second round',img_temp)
            # cv2.waitKey(1)

    # print ''

    img_temp = np.array(img_cv)
    for c in centroids:
        pt = (int(c[0]),int(c[1]))
        cv2.circle(img_temp, pt, 3, [0, 0, 255], 5)
    # cv2.imshow('post-filter',img_temp)
    # cv2.waitKey(1)

    img_to_show2 = img_temp

    img_to_show = np.hstack([np.asarray(img_to_show1), np.asarray(img_to_show2)])
    cv2.imshow('Before (Left) and After (Right) Noise Processing',img_to_show)
    cv2.waitKey(1)

    if len(centroids) < 8:
        print text_colors.WARNING + "Warning: Fewer than 8 points detected after filtering." + text_colors.ENDCOLOR
        #return (img_ret, found_feature, position_estimate, feature_points, rvecs, tvecs)
        return (img,False,None,None,None,None)



    #------------------------------------------------------------------------------#

    # Finding the Feature

    #------------------------------------------------------------------------------#

    # This section assigns the centroids that we found to the feature points.

    assigned_points = assign_points(centroids,np.array(img_cv))

    total_points_assigned = len(assigned_points[0]) + len(assigned_points[1])
    if not total_points_assigned == 8:
        print text_colors.WARNING + "Warning: Not exactly 8 points to assign." + text_colors.ENDCOLOR
        #return (img_ret, found_feature, position_estimate, feature_points, rvecs, tvecs)
        return (img,False,None,None,None,None)


    # With a full point correspondence, we can continue to the PnP solver and find 
    # the distance to our feature.



    #------------------------------------------------------------------------------#

    # Calculate Distance

    #------------------------------------------------------------------------------#

    mtx = np.array([[727.601951, 0.000000, 625.759658], \
        [0.000000, 731.883482, 470.335525], \
        [0.000000, 0.000000, 1.000000]])
    distortion = np.array([-0.279894, 0.056305, -0.004654, 0.001003, 0.000000])

    # Define object points. The units used here, [cm], will determine the units of the output.
    objp = np.zeros((8,1,3), np.float32)
    # Lowermost Subset
    objp[0] = [0.00, 0, 0]
    objp[1] = [15.9, 0, 0]
    objp[2] = [31.6, 0, 0]
    objp[3] = [47.5, 0, 0]
    # Uppermost Subset
    objp[4] = [0.00, 81, 25.6]
    objp[5] = [15.9, 81, 25.6]
    objp[6] = [31.6, 81, 25.6]
    objp[7] = [47.5, 81, 25.6]

    # Scale each input by a specific factor
    scale_factor_x = 1.0#32
    for i in range(0,len(objp)):
        objp[i][0] = objp[i][0]/scale_factor_x

    # Define feature points by determined correspondences
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
    if not debug_flag:
        img_morph = None
    
    return (img, True, Pw, feature_points, rvecs, tvecs)

class ImageProcessor(object):
    def __init__(self):
        # Previous Solution
        self.prev_solution = [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1,-1],[-1,-1,-1],False]

        # Image converter
        self.bridge = CvBridge()

        # Publishers
        self.pub_pos_as_str = rospy.Publisher("/my_pos_data",String,queue_size=1)
        self.pub_prev_sol = rospy.Publisher("/prev_solution",FeaturePoints,queue_size=1)
        self.pub_img_proc = rospy.Publisher("/camera/my_image_proc", Image, queue_size=1)

        # Subscribers
        self.sub_img_rect = rospy.Subscriber("rectified_image",Image,self.cbImgRect)
        self.sub_prev_sol = rospy.Subscriber("/prev_solution",FeaturePoints,self.cbPrevSol)

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

        outer_debug_flag = False
        if outer_debug_flag:
            pr = cProfile.Profile()
            pr.enable()

        # Convert the ROS image to an OpenCV image
        try:
            img = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            rospy.loginfo(e)

        # Obtain a distance estimation and an image with axes drawn on it
        img_ret, found_feature, position_estimate, feature_points, rvecs, tvecs = feature_extraction(img,self.prev_solution)

        # Convert the OpenCV image to a ROS image
        converted_successfully = True
        try:
            data = self.bridge.cv2_to_imgmsg(img_ret)
        except:
            converted_successfully = False
            # rospy.loginfo(e)

        prev_sol = FeaturePoints()
        prev_sol.found_solution = found_feature
        if found_feature:
            # Calculate distance
            x = position_estimate[0]
            y = position_estimate[1]
            z = position_estimate[2]
            distance_estimate = np.sqrt(x**2 + y**2 + z**2)

            # Set the solution for the next loop
            prev_sol.bottom_left_x         = feature_points[0][0][0]
            prev_sol.bottom_left_y         = feature_points[0][0][1]
            prev_sol.bottom_middle_left_x  = feature_points[1][0][0]
            prev_sol.bottom_middle_left_y  = feature_points[1][0][1]
            prev_sol.bottom_middle_right_x = feature_points[2][0][0]
            prev_sol.bottom_middle_right_y = feature_points[2][0][1]
            prev_sol.bottom_right_x        = feature_points[3][0][0]
            prev_sol.bottom_right_y        = feature_points[3][0][1]
            prev_sol.top_left_x            = feature_points[4][0][0]
            prev_sol.top_left_y            = feature_points[4][0][1]
            prev_sol.top_middle_left_x     = feature_points[5][0][0]
            prev_sol.top_middle_left_y     = feature_points[5][0][1]
            prev_sol.top_middle_right_x    = feature_points[6][0][0]
            prev_sol.top_middle_right_y    = feature_points[6][0][1]
            prev_sol.top_right_x           = feature_points[7][0][0]
            prev_sol.top_right_y           = feature_points[7][0][1]

            prev_sol.rvecs1 = rvecs[0]
            prev_sol.rvecs2 = rvecs[1]
            prev_sol.rvecs3 = rvecs[2]

            prev_sol.tvecs1 = tvecs[0]
            prev_sol.tvecs2 = tvecs[1]
            prev_sol.tvecs3 = tvecs[2]

            # Create a string for publishing
            data_str = ('[SUCCESS]\n\ttime:\n\t\tsecs: %s\n\t\tnsecs: %s\n' %(time.secs,time.nsecs))
            # data_str = ('[SUCCESS]\n\ttime:\n\t\tsecs: %s\n\t\tnsecs: %s\n\tposition' \
                # '\n\t\tx: %s\n\t\ty: %s\n\t\tz: %s\n' %(time.secs,time.nsecs,x,y,z))
        else:
            # Set the solution for the next loop
            prev_sol.bottom_left_x         = -1
            prev_sol.bottom_left_y         = -1
            prev_sol.bottom_middle_left_x  = -1
            prev_sol.bottom_middle_left_y  = -1
            prev_sol.bottom_middle_right_x = -1
            prev_sol.bottom_middle_right_y = -1
            prev_sol.bottom_right_x        = -1
            prev_sol.bottom_right_y        = -1
            prev_sol.top_left_x            = -1
            prev_sol.top_left_y            = -1
            prev_sol.top_middle_left_x     = -1
            prev_sol.top_middle_left_y     = -1
            prev_sol.top_middle_right_x    = -1
            prev_sol.top_middle_right_y    = -1
            prev_sol.top_right_x           = -1
            prev_sol.top_right_y           = -1

            prev_sol.rvecs1 = -1
            prev_sol.rvecs2 = -1
            prev_sol.rvecs3 = -1

            prev_sol.tvecs1 = -1
            prev_sol.tvecs2 = -1
            prev_sol.tvecs3 = -1

            # Create a string for publishing
            data_str = ('[FAILURE]\n\ttime:\n\t\tsecs: %s\n\t\tnsecs: %s\n' %(time.secs,time.nsecs))

        # Publish the image with axes drawn on it
        if converted_successfully:
            self.pub_img_proc.publish(data)

        # Publish the previous solution
        self.pub_prev_sol.publish(prev_sol)

        # Publish the position data as a string
        self.pub_pos_as_str.publish(data_str)

        if outer_debug_flag:
            pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print s.getvalue()


if __name__ == "__main__":
    # Initialize the node
    rospy.init_node('image_rect_listener')

    # Create the node
    node = ImageProcessor()

    # Spin
    rospy.spin()





































