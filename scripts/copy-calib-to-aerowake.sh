#!/bin/bash

VEHICLE=$1
DRIVETEST="$(lsblk)"
DATETIME=$(date +%Yy-%mm-%dd-%Hh-%Mm-%Ss)

if [ "$VEHICLE" = "" ]
then
        echo "Enter vehicle name"
else
  cp ~/creare_ws/src/creare/params/camera_calib.yaml ~/logs/camera/$VEHICLE-camcalib-$DATETIME.yaml
  echo "\n\033[0;34mSuccessfully copied camera calibration to ~/logs/camera/$VEHICLE-camcalib-$DATETIME.yaml\033[0m"
  # Check for crearedrive
  if ! echo "$DRIVETEST" | grep -q "crearedrive"
  then
          echo "Insert crearedrive"
  else
    cp ~/creare_ws/src/creare/params/camera_calib.yaml /crearedrive/camera/$VEHICLE-camcalib-$DATETIME.yaml
    echo "\n\033[0;34mSuccessfully copied camera calibration to /crearedrive/camera/$VEHICLE-camcalib-$DATETIME.yaml\033[0m"
  fi
fi
