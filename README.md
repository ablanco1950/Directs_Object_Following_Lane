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
=========

A program is incorporated that detects the lane on a low-performance video and in nighttime conditions and vertical trees, streetlights on the sides of the lane that can be detected as lines.

Furthermore, the video was not obtained from a vehicle following the lane, but rather intentionally moving away from and approaching the lane.

At times when the lane line is not detected, it stops and waits until it is detected again.

VIDEODirects_Object_Following_Lane-PoorVisibilityConditions.py

The success is achieved by adjusting the parameters corresponding to ROI, region of interest, and mask with thresold of the color to be detected (the white of the lane) and adding a filter based on cv2.hconcat in the process.

Similar results are obtained by executing:
VIDEODirects_Object_Following_Lane-PoorVisibilityConditions_filter2D.py
in which a treatment and an additional filter cv2.filter2d from [https://medium.com/practical-data-science-and-engineering](https://medium.com/practical-data-science-and-engineering/image-kernels-88162cb6585d) are added to the general procedure outlined in https://github.com/subin60/lane-detection /image-kernels-88162cb6585d

Worse results (fewer hit lines) and slower if executed:
VIDEODirects_Object_Following_Lane-PoorVisibilityConditions_FSRCNN.py
in  the last step uses Tensorflow implementation of 'Accelerating the Super-Resolution Convolutional Neural Network' (https://github.com/Saafke/FSRCNN_Tensorflow) from:

https://github.com/Saafke/FSRCNN_Tensorflow.


In case it's of interest, a program that cuts videos obtained on a mobile phone is also attached.

RecortaVideos.py 

Also, in case of interest, a simple light detector is attached.

Light detectorVIDEOContours.py

All programs produce a video as output:

demonstration.mp4 with the results

References:

https://github.com/subin60/lane-detection from where the treatment for lane detection has been obtained with some adaptations and simplifications. This treatment can clearly be improved and will be sought in subsequent editions. As of 01/01/2024, additional filters will be incorporated into this treatment that allow lane detection in cases of: poor quality videos, visibility, nearby trees... instead of the usual ones that are presented in lane detection projects.

https://github.com/alexstaravoitau/detecting-road-features from where the main  test video project_video.mp4 has been obtained.

https://github.com/alifiroozidev/lane-decection-sample-footages

https://github.com/sudharsan-007/opencv-lane-detection

https://github.com/ablanco1950/DetectCarDistanceAndRoadLane

https://github.com/ablanco1950/DetectSpeedLicensePlate_Yolov8_Filters_PaddleOCR

https://github.com/ablanco1950/DetectSpeedLicensePlate_RoboflowAPI_Filters_PaddleOCR

https://rollbar.com/blog/python-typeerror-nonetype-object-is-not-iterable/

http://elclubdelautodidacta.es/wp/2013/03/trigonometria-en-python/

https://levelup.gitconnected.com/opencv-python-reading-and-writing-images-and-videos-ed01669c660c

https://stackoverflow.com/questions/67302143/opencv-python-how-to-detect-filled-rectangular-shapes-on-picture

https://github.com/Saafke/FSRCNN_Tensorflow

https://medium.com/practical-data-science-and-engineering/image-kernels-88162cb6585d

