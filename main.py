import os, os.path
import sys #import system, which uses commands related to the windows system running
import cv2 #cv2 function
import math #math operations 
import glob #globbing utility.
import numpy as np 
from numpy import linalg
import python3_utils as utils
#import Stitcher_Method as stitch
import stitch_pictures as stitch
import get_sift_homography as get_sift_homography
import operator
from PIL import Image
from PIL import ImageDraw
import time
start_time = time.time()
    # Equalize Histogram of Color Images
def equalize_histogram_color(img):
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
	img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
	return img
i = 0
folder = ("C:\\Users\\SeanM\\Source\\Repos\\Stitcher_Method\\Stitching_Folder")
#folder = ("C:\\Users\\SeanM\\Source\\Repos\\Stitcher_Method\\Stitching_Folder\\Dense Reconstruction.nvm.cmvs\\00\\visualize")
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    print("Img: ",filename, "read.")
    #img = equalize_histogram_color(img)
    #print("Equalized")
    img = cv2.resize(img,None,fx=.25, fy=.25)
    print("Resized")
    #img = pX.blackRect(img)
    print("Stitching img: ", filename)
    if img is not None:
            if i==0:
                instance = img
                i = i+1
            else: 
                M =  get_sift_homography.get_sift_homography(equalize_histogram_color(instance), equalize_histogram_color(img))
                instance = stitch.stitch_image(img, instance, M)
current_time = (time.time() - start_time)
print("Current Runtime in seconds: ", current_time)
cv2.imwrite('All.jpg', instance) 
print("written to disk")

#i = 0

#folder = ("C:\\Users\\SeanM\\Source\\Repos\\imagestitcher_v1\\imagestitcher_v1\\Stitch_Folder")
#for filename in os.listdir(folder):
#    img = cv2.imread(os.path.join(folder,filename))
#    img = cv2.resize(img,None,fx=.25, fy=.25)
#    #img = pX.blackRect(img)
#    print("Stitching img: ", filename)
#    if img is not None:
#            if i==0:
#                instance = img
#                i = i+1
#            else: 
#                instance = stitch.stitch(instance, img, 0)

#cv2.imwrite('All.jpg', instance) 
#print("written to disk")

