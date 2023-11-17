# -*- coding: utf-8 -*-


import cv2
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
lenthRegion=4.5 #the depth of the considered region corresponds
                # to the length of a parking space which is usually 4.5m

                
#
# adapted and simplified from
# https://github.com/subin60/lane-detection
# https://medium.com/@subin60/detecting-lane-lines-using-computer-vision-techniques-in-python-a-hands-on-experience-badcc6f01933
#
def process_frame(image):
    # Convert the input image to HLS color space
    #image1=image.resize((1280,720))
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    #image_hls=image_hls.resize((1280,720))
    # Define color range for white mask
    lower_threshold = np.uint8([0, 200, 0])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image_hls, lower_threshold, upper_threshold)

    # Define color range for yellow mask
    lower_threshold = np.uint8([10, 0, 100])
    upper_threshold = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(image_hls, lower_threshold, upper_threshold)

    # Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Apply the mask to the input image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    #cv2.imshow("masked_image", masked_image)
    #cv2.waitKey(0)

    # Convert the masked image to grayscale
    masked_image_gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to the grayscale image
    masked_image_gray_blur = cv2.GaussianBlur(masked_image_gray, (13, 13), 0)

    # Apply Canny edge detection to the blurred image
    masked_image_gray_blur_edge_detec = cv2.Canny(masked_image_gray_blur, 50, 150)
    #cv2.imshow(" edge_detec",  masked_image_gray_blur_edge_detec)
    #cv2.waitKey(0)
    

    # Create a region of interest mask
    mask = np.zeros_like(masked_image_gray_blur_edge_detec)
    channel_count = image.shape[2]
    ignore_mask_color = (255,) * channel_count
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    mask = cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Apply the region of interest mask to the edge detected image
    masked_image = cv2.bitwise_and(masked_image_gray_blur_edge_detec, mask)

    #cv2.imshow("masked_image_gray_blur",  masked_image)
    #cv2.waitKey(0)

    # Apply Hough transform to the masked image
    hough_lines = cv2.HoughLinesP(masked_image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

    # Draw Hough lines on a copy of the input image
    image_co = np.copy(image)
    #solo se va a resaltar la linea mayor
    #for line in hough_lines:
    #    for x1, y1, x2, y2 in line:
    #        cv2.line(image_co, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #cv2.imshow("image_co",  image_co)
    #cv2.waitKey(0)
       
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
           
            #length = np.sqrt(((y1 - y2) ** 2) + ((x1 - x2) ** 2))  # Calculate the length of the line
            # http://elclubdelautodidacta.es/wp/2013/03/trigonometria-en-python/
            length= math.hypot((x1 - x2),(y1 - y2))
            #print("x1=" + str(x1)+ " y1=" + str(y1) + "x2=" + str(x2)+ " y2=" + str(y2))
            if length > lengthMax:
                lengthMax = length
                
                x1max=x1
                y1max=y1
                x2max=x2
                y2max=y2
                mmax=m
                bmax=b
                
               
            
    #cv2.line(image_co, (x1max, y1max), (x2max, y2max), (0, 255, 0), 2)
    #print("x1=" + str(x1max)+ " y1=" + str(y1max) + " x2=" + str(x2max)+ " y2=" + str(y2max) + " m=" + str(m)+ " b=" + str(b))
    #cv2.imshow("image_co",  image_co)
    #cv2.waitKey(0)        
    return image_co, x1max, y1max, x2max, y2max, mmax, bmax
                         
  
import os
import cv2
import numpy as np
from glob import glob


def OptionVideo(dirVideo):
    cap = cv2.VideoCapture(dirVideo)
    # https://levelup.gitconnected.com/opencv-python-reading-and-writing-images-and-videos-ed01669c660c
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps=5.0
    frame_width = 680
    frame_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
     
    video_writer = cv2.VideoWriter('demonstration.mp4',fourcc,fps, size) 
    ContFrames=0
    ContFramesJumped=0
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
            #image, x1, y1, x2, y2, m, b = process_image_video(img)
            image, x1, y1, x2, y2, m, b  = process_frame(img)
            height = image.shape[0]
            width = image.shape[1]
            #print("  height=" +str( height)+ " width=" + str(width))

            if y1 > y2:
                Xtarget=x2 + 50
                Ytarget=y2
                
            else:
                Xtarget=x1+ 50
                Ytarget=y1
            
              
            cv2.circle(image,(int(Xtarget), int(Ytarget)), 10, (255,0,0), thickness=5)
            
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
    
# from https://github.com/alexstaravoitau/detecting-road-features
         
#dirVideo="C:\\detecting-road-features-master\\detecting-road-features-master\\data\\video\\project_video.mp4"
dirVideo="project_video.mp4"
OptionVideo(dirVideo)
