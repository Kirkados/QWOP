#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 20:37:51 2019

@author: StephaneMagnan
"""
#import ffmpeg
#
#
#./ffmpeg -framerate 10 -i "frames/10.0/frame_%05d.png" -vcodec mpeg4 -y movie.mp4

#for filename in os.listdir(path):
#    if (filename.endswith(".mp4")): #or .avi, .mpeg, whatever.
#        os.system("ffmpeg -i {0} -f image2 -vf fps=fps=1 output%d.png".format(filename))
#    else:
#        continue

import cv2
import numpy as np
import os
 
from os.path import isfile, join
 
def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
    for f in files:
        if f[-4:] != ".png":
            files.remove(f)
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[-9:-4]))
 
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
        os.remove(filename)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
 
def main():
    pathIn= './frames/test_rec/'
    pathOut = './frames/test_rec/video.avi'
    fps = 5
    convert_frames_to_video(pathIn, pathOut, fps)
 
if __name__=="__main__":
    main()
