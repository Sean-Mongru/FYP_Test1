import os, os.path
import sys #import system, which uses commands related to the windows system running
import cv2 #cv2 function
import math #math operations 
import glob #globbing utility.
import numpy as np 
from numpy import linalg
import python3_utils as utils
import time
#import Stitcher_Method as stitch
import gc #Garbage Collector

# Find SIFT and return Homography Matrix
def get_sift_homography(img1, img2):
	# Initialize SIFT 
    #sift = cv2.xfeatures2d.SIFT_create()
    #orb = cv2.ORB_create()
    surf = cv2.xfeatures2d.SURF_create()

    gray1 =  cv2.cvtColor(img1 ,cv2.COLOR_BGR2GRAY)
    base_img_1 = cv2.GaussianBlur(gray1,(5,5),0)
    gray2 =  cv2.cvtColor(img2 ,cv2.COLOR_BGR2GRAY)
    base_img_2 = cv2.GaussianBlur(gray2,(5,5),0)

	# SIFT Extract keypoints and descriptors
    #k1, d1 = sift.detectAndCompute(base_img_1, None)
    #k2, d2 = sift.detectAndCompute(base_img_2, None)

	# SURF Extract keypoints and descriptors
    k1, d1 = surf.detectAndCompute(base_img_1, None)
    k2, d2 = surf.detectAndCompute(base_img_2, None)

    # ORB Extract keypoints and descriptors
    #k1, d1 = orb.detectAndCompute(base_img_1, None)
    #k2, d2 = orb.detectAndCompute(base_img_2, None)

	# Bruteforce matcher on the descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1,d2, k=2)

	# Make sure that the matches are good
    verify_ratio = 0.8 # Source: stackoverflow
    verified_matches = []
    for m1,m2 in matches:
		# Add to array only if it's a good match
        if m1.distance < 0.8 * m2.distance:
            verified_matches.append(m1)

	# Mimnum number of matches
    min_matches = 8
    if len(verified_matches) > min_matches:
		
		# Array to store matching points
        img1_pts = []
        img2_pts = []

		# Add matching points to array
        for match in verified_matches:
            img1_pts.append(k1[match.queryIdx].pt)
            img2_pts.append(k2[match.trainIdx].pt)
        img1_pts = np.float32(img1_pts).reshape(-1,1,2)
        img2_pts = np.float32(img2_pts).reshape(-1,1,2)
		
		# Compute homography matrix
        M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
        return M
    else:
        print ("Error: Not enough matches")
        exit()