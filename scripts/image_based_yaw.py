#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Int16

# Set camera parameters in camera_params.yaml

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

def get_yaw_cmd(pix_dist, function_fit = 'linear'):
	'''
	This function returns the number of degrees (CW positive) that the camera should turn to place the feature at the center of its image. 
	
	The function calculates the necessary change in yaw according to a specified function, linear or quadratic. The yaw_command returned is proportional to 

	The maximum yaw is somewhat arbitrarily chosen since the camera will operate at significantly variable distances from the target. The maximum yaw is chosen here to be 15 degrees. 

	Note:
		pix_dist > 0 -> turn CW
		pix_dist < 0 -> turn CCW
	'''

	pix_dist_mag = np.abs(pix_dist)
	a = 0.00003662109 # maximum yaw divided by image width in pixels squared
	b = 0.0234375 # maximum yaw divided by image width in pixels

	if function_fit == 'quadratic':
		yaw_cmd_mag = a*pix_dist_mag**2
	elif function_fit == 'linear':
		yaw_cmd_mag = b*pix_dist_mag
	else:
		raise TypeError

	yaw_cmd_mag = np.floor(yaw_cmd_mag)

	if pix_dist > 0:
		yaw_cmd = -yaw_cmd_mag
	else:
		yaw_cmd = yaw_cmd_mag

	return yaw_cmd

def get_yaw_deg(img):
	'''
	This function takes in an image and returns an integer value in degrees of the desired differential yaw of the vehicle. 
	'''
	
	# cv2.imshow('orig',img)
	# cv2.waitKey(1)

	# # Blur the image
	# kernelX = np.array([0.3,0.4,0.3])
	# kernelY = np.array([0.3,0.4,0.3])
	# img = cv2.sepFilter2D(img,-1,kernelX,kernelY)

	# cv2.imshow('blur',img)
	# cv2.waitKey(1)

	# Convert to binary
	max_value = 255
	block_size = 5
	const = 1
	threshold_value = 100
	_,img = cv2.threshold(img,threshold_value,max_value,cv2.THRESH_BINARY)

	# cv2.imshow('binary',img)
	# cv2.waitKey(1)

	# Morph image to 'close' the shapes that are found
	kernel = np.ones((2,2),np.uint8)
	img = cv2.dilate(img,kernel,iterations = 1)
	img = cv2.erode(img,kernel,iterations = 1)

	# cv2.imshow('morph',img)
	# cv2.waitKey(1)

	# Remove pixels on very edge of image (camera driver places white pixels at top left of image for some reason)
        # edge_buffer = 1
        # img = img[edge_buffer:img.shape[0] - edge_buffer, edge_buffer:img.shape[1] - edge_buffer]

	# Find contours
	contours, _ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	#_, contours, _ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

	# img_temp = np.array(img)
	# cv2.drawContours(img_temp, contours, -1, (255,255,255), 3)
	# cv2.imshow('draw contours',img_temp)
	# cv2.waitKey(1)

	# Extract centroids
	centroids = []
	all_centers = set([])
	nearby_restriction_value = 5
	for single_contour in contours:
		# Obtain the coordinates of the center of the contour
		(x,y),_ = cv2.minEnclosingCircle(single_contour)
		center = (int(x),int(y))
		# If we have already detected a point at a given coordinate, do not add it again. Since the center is cast to an integer, this step ensures that redundant points are not counted. e.g. (617.5,742.2) and (617.9,742.4) may be detected as separate points, but clearly they refer to the same point in the image and only one should be counted. 
		if center not in all_centers:
			all_centers.add(center)
			# Add nearby-restriction so that we do not get redundant centroids
			for i in range(-nearby_restriction_value,nearby_restriction_value + 1):
				for j in range(-nearby_restriction_value,nearby_restriction_value + 1):
					nearby_center = (center[0] + i,center[1] + j)
					all_centers.add(nearby_center)
			centroids.append((x,y))

	if len(centroids) > 0:
		'''
		# Draw detected centroids as circles
		img_temp = np.array(img)
		for cntr in centroids:
		    center = (int(cntr[0]),int(cntr[1]))
		    cv2.circle(img_temp, center, 3, 255, 3)
		# cv2.imshow('obt init cntds',img_temp)
		# cv2.waitKey(1)
		'''

		# Calculate the overall centroid of the detected features
		x_mean = np.mean([coord[0] for coord in centroids])
		y_mean = np.mean([coord[1] for coord in centroids])

		'''
		# Draw a grey square at the overall centroid location
		img_temp = np.array(img)
		half_length = 3
		pt1 = (int(x_mean - half_length), int(y_mean - half_length))
		pt2 = (int(x_mean + half_length), int(y_mean + half_length))
		cv2.rectangle(img_temp, pt1, pt2, 150, 1)
		# cv2.imshow('overall centroid as square',img_temp)
		# cv2.waitKey(1)
		'''

		# Get image center
		x_img_center = int(img.shape[1]/2)
		y_img_center = int(img.shape[0]/2)

		'''
		# Draw crosshairs at center of image
		img_temp = np.array(img)
		half_length = 10
		pt1 = (x_img_center, y_img_center - half_length)
		pt2 = (x_img_center, y_img_center + half_length)
		pt3 = (x_img_center - half_length, y_img_center)
		pt4 = (x_img_center + half_length, y_img_center)
		cv2.rectangle(img_temp, pt1, pt2, 150, 1)
		cv2.rectangle(img_temp, pt3, pt4, 150, 1)
		# cv2.imshow('center of image as cross',img_temp)
		# cv2.waitKey(1)
		'''

		# Calculate pixel distance between feature centroid and center of image
		pix_dist = int(x_img_center - x_mean)

		'''
		# Draw a line of pix_dist pixels to show the distance calculated
		pt1 = (x_img_center, y_img_center)
		pt2 = (x_img_center - pix_dist, y_img_center)
		cv2.line(img_temp,pt1,pt2,150,1)
		# cv2.imshow('distance to center',img_temp)
		# cv2.waitKey(1)
		'''

		# Determine the yaw command based upon the distance calculated
		yaw_command = get_yaw_cmd(pix_dist, function_fit = 'linear')

		'''
		# Print the calculated yaw onto the image
		font = cv2.FONT_HERSHEY_SIMPLEX
		txt_str = '%d' %(yaw_command)
		#cv2.putText(img_temp,txt_str,(100,100), font, 2, 255, 2, cv2.LINE_AA)
		# cv2.imshow('yaw_cmd',img_temp)
		# cv2.waitKey(1)
		'''

		# Return the yaw command
		return yaw_command
	else:
		# Command no-change in yaw if there are no lights in the image
		return 0

class YawCommandCreator(object):
	def __init__(self):
		# Image converter
		self.bridge = CvBridge()

		# Publishers
		self.pub_yaw_deg = rospy.Publisher("/yaw_deg",Int16,queue_size=1)

		# Subscribers
		self.sub_img_raw  = rospy.Subscriber("raw_image",Image,self.cbImgRaw)

	def cbImgRaw(self,data):
		# Collect time of when image was received
		time = rospy.get_rostime()

		# Convert the ROS image to an OpenCV image
		try:
			img = self.bridge.imgmsg_to_cv2(data, '8UC1')
		except CvBridgeError:
			print text_colors.WARNING + 'Warning: Image converted unsuccessfully before processing.' + text_colors.ENDCOLOR
			#rospy.loginfo(e)

		# Remove the edge of the image since there are erroneous pixels in the top left of the image (some sort 
		# of image-type conversion error perhaps?)
		edge_buffer = 1
		img = img[edge_buffer:img.shape[0] - edge_buffer, edge_buffer:img.shape[1] - edge_buffer]

		# Perform yaw calculation
		yaw_cmd = get_yaw_deg(img)

		# Publish the yaw command as an integer
		self.pub_yaw_deg.publish(yaw_cmd)

if __name__ == "__main__":
	# Initialize the node
	rospy.init_node('image_yaw_creator_node')

	# Create the yaw command node
	node = YawCommandCreator()

	# Spin
	rospy.spin()

