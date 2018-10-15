#!/bin/bash

DATETIME=$(date +%Yy-%mm-%dd-%Hh-%Mm-%Ss)

cp ~/creare_ws/src/creare/params/camera_calib.yaml ~/logs/camera/camera_calib_$DATETIME.yaml
cp ~/creare_ws/src/creare/params/camera_calib.yaml /crearedrive/camera/camera_calib_$DATETIME.yaml

