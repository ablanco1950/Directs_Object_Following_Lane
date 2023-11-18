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

Notes:

The project can clearly be improved.

Like all video monitoring projects of a highway lane, it is favored because the camera follows the driver's movements, which practically ensures lane following.

In this project, the driver's lane is not followed but rather the adjacent one to the right, but as the lane detection is based on the treatment developed in the project https://github.com/subin60/lane-detection with some adaptations and simplifications , there is a favor in the detection of this lane that the driver is following.


References:

https://github.com/subin60/lane-detection from where the treatment for lane detection has been obtained with some adaptations and simplifications. This treatment can clearly be improved and will be sought in subsequent editions.

https://github.com/alexstaravoitau/detecting-road-features from where the test video has been obtained.

https://github.com/sudharsan-007/opencv-lane-detection


https://github.com/ablanco1950/DetectCarDistanceAndRoadLane

https://github.com/ablanco1950/DetectSpeedLicensePlate_Yolov8_Filters_PaddleOCR

https://github.com/ablanco1950/DetectSpeedLicensePlate_RoboflowAPI_Filters_PaddleOCR

