# -*- coding: utf-8 -*-
# 

dirVideo="VID_PoorVisiBility.mp4"

import cv2
import numpy as np
import math
import time
TimeIni=time.time()
# in  14 minutes = 800 seconds finish  
TimeLimit=800
               
#

import cv2
import numpy as np


def OptionVideo(dirVideo):
    cap = cv2.VideoCapture(dirVideo)
    # https://levelup.gitconnected.com/opencv-python-reading-and-writing-images-and-videos-ed01669c660c
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps=5.0
    
    frame_width = 720
    frame_height = 1280
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
     
    video_writer = cv2.VideoWriter('demonstration.mp4',fourcc,fps, size) 
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
            #img = cv2.flip(img,0)
            
            # https://stackoverflow.com/questions/67302143/opencv-python-how-to-detect-filled-rectangular-shapes-on-picture
            #
            
            image=img
           
            # grayscale
            result = image.copy()
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

            # adaptive threshold
            thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,9)

            # Fill rectangular contours
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(thresh, [c], -1, (255,255,255), -1)

            # Morph open
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

            # Draw rectangles, the 'area_treshold' value was determined empirically
            cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            area_treshold = 400
            for c in cnts:
                if cv2.contourArea(c) > area_treshold :
                  x,y,w,h = cv2.boundingRect(c)
                  cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)

            
            cv2.imshow('image', image)
            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'): break 
            # saving video
            video_writer.write(image)    
            # a los 10 minutos = 600 segundos acaba     
            if time.time() - TimeIni > TimeLimit:
                    
                    break
                   
            if ContFrames > 4 :
                cv2.destroyAllWindows()
                ContFrames =1
            
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    

OptionVideo(dirVideo)
