#!/usr/bin/env python

import numpy as np
import sys
import time

def raise_exception():
  raise Exception('oops')

def parse_line(line):
  '''
  Parses a string for desired data.
  '''
  if 'ATTITUDE' in line:
    line = line.split(',')
    t = line[0]
    r = float(line[5].split(':')[1].replace(' ',''))*180/np.pi
    p = float(line[6].split(':')[1].replace(' ',''))*180/np.pi
    y = float(line[7].split(':')[1].replace(' ',''))*180/np.pi
    return t,r,p,y
  else:
    return None

if __name__ == "__main__":

  # Set filename
  logpath = sys.argv[1]
  logdir = logpath[:logpath.rfind('/') + 1]
  logname = logpath[logpath.rfind('/') + 1:]
  savename = logname.replace('uav','uav-att').replace('gcs','gcs-att').replace('.log','.csv')

  # Open output file and input file 
  with open(logdir + savename,'w') as wf:
    with open(logdir + logname, 'r') as rf:

      # Read all data into a variable
      all_data = rf.readlines()

      # Write file header
      wf.write('t[s.ns],roll[deg],pitch[deg],yaw[deg],%s\n' %(logname))

      # Loop through each line and parse data
      for line in all_data:
        data = parse_line(line)
        if not data is None:
          wf.write('%s,%s,%s,%s\n' %(data))
