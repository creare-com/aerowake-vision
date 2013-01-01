#!/bin/bash

DATETIME=$(date +%Yy-%mm-%dd-%Hh-%Mm-%Ss)

cp ~/.ros/camera_info/0.yaml ~/creare_ws/src/creare/params/camera_calib.yaml
cp ~/.ros/camera_info/0.yaml ~/logs/camera/camera_calib_$DATETIME.yaml
