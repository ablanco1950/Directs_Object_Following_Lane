# Directs_Object_Following_Lane
Project that positions an object in a video following a road lane.

Requirements:
All the files that accompany the project and packages that, if missing, can be installed with a simple pip

import numpy as np

import cv2

import time

import math


Execute:

VIDEODirects_Object_Following_Lane.py

In the video you can see an object marked with a blue circle forced to follow the second lane to the left.

Also, you can test the program with the video solidYellowLeft.mp4 (downloaded from https://github.com/alifiroozidev/lane-decection-sample-footages), removing the # that cancels instruction 7. In both cases it is about placing the object that follows in the second rail on the left.

To test a case of following the first lane to the left, you must change instruction 3 by setting OptionLane = 1 and removing the # that cancels instruction 12 and allows you to test this case with the video solidWhiteRight.mp4 (downloaded from from https://github.com/alifiroozidev/lane-decection-sample-footages).
By removing the # that cancels instruction 13, you can check it with the video road_-_28287 (540p).mp4

Notes:

The project can clearly be improved.

Like all video monitoring projects of a highway lane, it is favored because the camera follows the driver's movements, which practically ensures lane following.

In this project, the driver's lane is not followed but rather the adjacent one to the right, but as the lane detection is based on the treatment developed in the project https://github.com/subin60/lane-detection with some adaptations and simplifications , there is a favor in the detection of this lane that the driver is following.

01/01/2024
A program is incorporated that detects the lane on a low-performance video and in nighttime conditions and vertical trees, streetlights on the sides of the lane that can be detected as lines:
VIDEODirects_Object_Following_Lane-PoorVisibilityConditions.py
The success is achieved by adjusting the parameters corresponding to ROI, region of interest, and mask with thresold of the color to be detected (the white of the lane) and adding a filter based on cv2.hconcat in the process.


References:

https://github.com/subin60/lane-detection from where the treatment for lane detection has been obtained with some adaptations and simplifications. This treatment can clearly be improved and will be sought in subsequent editions.

https://github.com/alexstaravoitau/detecting-road-features from where the main  test video project_video.mp4 has been obtained.

https://github.com/alifiroozidev/lane-decection-sample-footages

https://github.com/sudharsan-007/opencv-lane-detection

https://github.com/ablanco1950/DetectCarDistanceAndRoadLane

https://github.com/ablanco1950/DetectSpeedLicensePlate_Yolov8_Filters_PaddleOCR

https://github.com/ablanco1950/DetectSpeedLicensePlate_RoboflowAPI_Filters_PaddleOCR

