#!/usr/bin/env python2

import numpy as np
import sys

def calculate_new_coords(brg,orig_coords):
  '''
  Rotates each coordinate in 2D space by the specified bearing. Altitude (Down) is unchanged. 
  '''

  # Extract arrays
  N = orig_coords[0]
  E = orig_coords[1]
  D = orig_coords[2]

  # Convert to radians
  brg = float(brg)
  brg = brg*np.pi/180

  # Create rotation matrix
  R = [[np.cos(brg),-np.sin(brg)],[np.sin(brg),np.cos(brg)]]
  R = np.matrix(R)

  # Rotate NE parts and assign to new array
  new_coords = [[],[],list(D)]
  for i in range(0,len(N)):
    arr = np.array([[N[i]],[E[i]]])
    new_N,new_E = R*arr
    new_N = round(new_N,5)
    new_E = round(new_E,5)
    new_coords[0].append(new_N)
    new_coords[1].append(new_E)

  return new_coords

if __name__ == "__main__":

  # Set global variables
  logpath = sys.argv[1]
  logdir = logpath[:logpath.rfind('/') + 1]
  logname = logpath[logpath.rfind('/') + 1:]
  savename = logname.replace('bag-pose','bag-pose-ned')

  # Parse user input
  if len(sys.argv) == 3:
    bearing = sys.argv[2]
  else:
    raise Exception('Usage: python ship_pose_to_ned_pose.py <full path> <bearing>')

  with open(logdir + savename,'w') as wf:
    with open(logdir + logname,'r') as rf:

      # Read all data into a variable
      all_lines = rf.readlines()

      # Write file header
      wf.write('time[s.ns],elapsed[s.ns],x[m],y[m],z[m],yaw[deg],pitch[deg],roll[deg],rx[deg],ry[deg],rz[deg],N[m],E[m],D[m],%s\n' %(logname))

      # Get gcs coords
      x = [float(line.split(',')[2]) for line in all_lines[1:]]
      y = [float(line.split(',')[3]) for line in all_lines[1:]]
      z = [float(line.split(',')[4]) for line in all_lines[1:]]

      # Get ned coords
      gcs_coords = [x,y,z]
      ned_coords = calculate_new_coords(bearing,gcs_coords)

      # Remove newline char
      all_lines = [line[:-1] for line in all_lines]

      # Append ned coords to each line and write to file
      i = 0
      for j in range(1,len(all_lines)):
        n = ned_coords[0][i]
        e = ned_coords[1][i]
        d = ned_coords[2][i]
        write_str = all_lines[j] + ',%s,%s,%s\n' %(n,e,d)
        i += 1
        wf.write(write_str)


