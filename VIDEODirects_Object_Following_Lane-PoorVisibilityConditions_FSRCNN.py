# -*- coding: utf-8 -*-
# 


dirVideo="VID_PoorVisiBility.mp4"

import cv2
# suggested by Wilbur
# https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models
# https://learnopencv.com/super-resolution-in-opencv/#sec5
# https://learnopencv.com/super-resolution-in-opencv/
ocv_model = cv2.dnn_superres.DnnSuperResImpl_create()
ocv_weight = 'FSRCNN_x4.pb'
ocv_model.readModel(ocv_weight)
ocv_model.setModel('fsrcnn', 4)
import numpy as np
import math
import time
TimeIni=time.time()
# in  14 minutes = 800 seconds finish  
TimeLimit=800
# Max number of Snapshots to consider a image
LimitSnapshot=1

# to increase the speed of the process,
# even if some license plates are lost,
# only one snapshot out of every SpeedUpFrames is processed
SpeedUpFrames=5
# to increase speed, jump frames  
ContFramesJumped=0
fps=25 #frames per second of video dirvideo, see its properties
fpsReal= fps/SpeedUpFrames # To speed up the process only one of SpeedUpFrames
                           # is considered


TotalLineHits=0


def process_frame(image, TotalLineHits):
    
    #Convert the input image to HLS color space
    
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # critical parameters
   
    lower_threshold = np.uint8([110, 110, 110])    
    upper_threshold = np.uint8([245, 245, 245])
    
    white_mask = cv2.inRange(image_hls, lower_threshold, upper_threshold)
   
    # Apply the mask to the input image
    masked_image = cv2.bitwise_and(image, image, mask=white_mask)
    #cv2.imshow("masked_image", masked_image)
    #cv2.waitKey(0)

    # Apply Gaussian blur to the grayscale image
    
    masked_image_gray_blur = cv2.GaussianBlur(masked_image, (13, 13), 0)
    
    # Apply Canny edge detection to the blurred image
    masked_image_gray_blur_edge_detec = cv2.Canny(masked_image_gray_blur, 50, 150)
   
    #cv2.imshow(" edge_detec",  masked_image_gray_blur_edge_detec)
    #cv2.waitKey(0)
    
    # Create a region of interest mask
    mask = np.zeros_like(masked_image_gray_blur_edge_detec)
    channel_count = image.shape[2]
    ignore_mask_color = (255,) * channel_count
    
    rows, cols = image.shape[:2]

    # critical parameters for ROI
    bottom_left = [cols * 0.2, rows * 0.45]
    top_left = [cols * 0.2, rows * 0.3]
    bottom_right = [cols * 0.50, rows * 0.45]
    top_right = [cols * 0.50, rows * 0.3]
    
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    mask = cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    # Apply the region of interest mask to the edge detected image
    masked_image = cv2.bitwise_and(masked_image_gray_blur_edge_detec, mask)
    
   
    gray= ocv_model.upsample(masked_image)
    gray=cv2.resize(gray, (720, 1280)) 
    
    hough_lines = cv2.HoughLinesP(gray, rho=1, theta=np.pi/180, threshold=50, minLineLength=20, maxLineGap=300)
   
    image_co = np.copy(image)
    

    #cv2.imshow("image_co",  image_co)
    #cv2.waitKey(0)
    
    # https://rollbar.com/blog/python-typeerror-nonetype-object-is-not-iterable/
    if hough_lines is None:
        return image_co, 0, 0, 0, 0, 0, 0, 0,TotalLineHits
    
       
    for line in hough_lines:
        
        lengthMax=0
        
        x1max=0
        y1max=0
        x2max=0
        y2max=0
        mmax=0.0
        bmax=0.0
      
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            # line y=mx+b
            #

                      
            m = (y1 - y2) / (x1 - x2)
            b = y1 - (m * x1)

            # Calculate the length of the line
            # http://elclubdelautodidacta.es/wp/2013/03/trigonometria-en-python/
            length= math.hypot((x1 - x2),(y1 - y2))
            #if length > 1000: continue
            if length > lengthMax:
                
                
                lengthMax = length
                
                x1max=x1
                y1max=y1
                x2max=x2
                y2max=y2
                mmax=m
                bmax=b
                
               
            
    cv2.line(image_co, (x1max, y1max), (x2max, y2max), (0, 255, 0), 2)
    if x1max !=0:
             TotalLineHits=TotalLineHits+1
    #print("x1=" + str(x1max)+ " y1=" + str(y1max) + " x2=" + str(x2max)+ " y2=" + str(y2max) + " m=" + str(m)+ " b=" + str(b))
    #cv2.imshow("image_co",  image_co)
    #cv2.waitKey(0)
    #print(mmax)
    return image_co, x1max, y1max, x2max, y2max, mmax, bmax,lengthMax, TotalLineHits
                         
  

import cv2
import numpy as np


def OptionVideo(dirVideo, TotalLineHits):
    cap = cv2.VideoCapture(dirVideo)
    # https://levelup.gitconnected.com/opencv-python-reading-and-writing-images-and-videos-ed01669c660c
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps=5.0
    
    # Videos from camera of a cheap movil
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
            """
            ContFramesJumped=ContFramesJumped+1
            if ContFramesJumped < SpeedUpFrames:
               continue
            else:
               ContFramesJumped=0
            """
                       
            # caso del video de mi movil que
            # que aparecian las imagenes invertidas
            
            #img = cv2.flip(img,0)
            
            image, x1, y1, x2, y2, m, b , lengthMax, TotalLineHits = process_frame(img, TotalLineHits)
            if x1== 0: continue  # for "solidWhiteRight.mp4" 
            #print(lengthMax)          
                        
            height = image.shape[0]
            width = image.shape[1]
            #print("  height=" +str( height)+ " width=" + str(width))

            if y1 > y2:
                #Xtarget=x2 + 20
                Xtarget=x2 
                Ytarget=y2
                
            else:
                #Xtarget=x1+ 20
                Xtarget=x1
                Ytarget=y1
            
              
            cv2.circle(image,(int(Xtarget), int(Ytarget)), 20, (255,0,0), thickness=5)
            print( " X target =" + str(int(Xtarget))+" Y target =" + str(int(Ytarget)))
            cv2.imshow('Frame', image)
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
    print("Lines hit=" +str(TotalLineHits))

OptionVideo(dirVideo, TotalLineHits)
