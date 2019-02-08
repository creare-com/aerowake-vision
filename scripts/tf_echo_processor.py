#!/usr/bin/env python

import numpy as np
import re
import sys
import time

def raise_exception():
  raise Exception('oops')

def parse_line(line):
  '''
  Parses a string for desired data. Input lines are expected to be in the exact format as the output of 'rosrun tf tf_echo /body1 /body2'.
  '''
  if 'time' in line:
    line = line.split(' ')
    time = line[-1][:-1]
    return 't',time
  elif 'Translation' in line:
    line = line.split(' ')
    x = line[-3].replace('[','').replace(',','')
    y = line[-2].replace(',','')
    z = line[-1].replace(']','').replace('\n','')
    return 'p',x,y,z
  elif 'degree' in line:
    line = line.split(' ')
    x = line[-3].replace('[','').replace(',','')
    y = line[-2].replace(',','')
    z = line[-1].replace(']','').replace('\n','')
    return 'o',x,y,z
  else:
    return None

if __name__ == "__main__":

  # Set filename
  tfpath = sys.argv[1]
  tfdir = tfpath[:tfpath.rfind('/') + 1]
  tfname = tfpath[tfpath.rfind('/') + 1:]
  filename = tfname.replace('tfe','tfe-proc').replace('.txt','.csv')

  # Open output file and input file
  with open(tfdir + filename,'w') as wf:
    with open(tfdir + tfname, 'r') as rf:

      # Read all data into a variable
      all_data = rf.readlines()

      # Set initial loop variables
      write_str = ''
      t_prev = 'not in write_str for first loop'
      write_to_file = False

      # Write file header
      wf.write('t,x,y,z,rx,ry,rz,%s\n' %(tfname))

      # Loop through each line and parse data
      for line in all_data:
        data = parse_line(line)
        if not data is None:
          if data[0] == 't':
            t = data[1]
            write_str = write_str + t + ','
          elif data[0] == 'p':
            x,y,z = data[1:]
            write_str = write_str + x + ',' + y + ',' + z + ','
          elif data[0] == 'o':
            rx,ry,rz = data[1:]
            write_str = write_str + rx + ',' + ry + ',' + rz + '\n'
            write_to_file = True
          else:
            pass

        # Write to file once we have time, position, and orientation
        if write_to_file:
          # Prevent multiple points at the exact same time (since tf_echo doesn't provide enough significant figures in nanoseconds)
          if not t_prev in write_str:
            wf.write(write_str)
          t_prev = write_str[:14]
          write_str = ''
          write_to_file = False
