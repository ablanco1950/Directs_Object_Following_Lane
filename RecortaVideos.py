# -*- coding: utf-8 -*-
# 

dirVideo="VID_20231126_192925.mp4"

import cv2
import numpy as np
fps=5.0

cap = cv2.VideoCapture(dirVideo)

# Videos from camera of a cheap movil
frame_width = 720
frame_height = 1280
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
 


# https://levelup.gitconnected.com/opencv-python-reading-and-writing-images-and-videos-ed01669c660c
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
 
video_writer = cv2.VideoWriter('VID_PoorVisiBility.mp4',fourcc,fps, size) 
ContFrames=0
ContFramesJumped=0
SwIni=0
x1Ant=0
while (cap.isOpened()):
    ret, img = cap.read()
    if ret != True: break
    
    else:
        # caso del video de mi movil que
        # que aparecian las imagenes invertidas
            
        img = cv2.flip(img,0)

        ContFrames=ContFrames+1
        if ContFrames > 700 : break
        #cv2.imshow('Frame', img)
        # Press Q on keyboard to exit
        #if cv2.waitKey(25) & 0xFF == ord('q'): break 
        # saving video
        video_writer.write(img)    
        
cap.release()
video_writer.release()
cv2.destroyAllWindows()
print("ContFrames = " +str(ContFrames))

