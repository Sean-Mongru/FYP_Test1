import os, os.path
 #os functions (basically can change and manipulate directories
import sys #import system, which uses commands related to the windows system running
import cv2 #cv2 function
import math #math operations 
import glob #globbing utility.
import numpy as np 
from numpy import linalg
import python3_utils as utils

def show_img(img):
    cv2.imshow(img, img)
    cv2.waitKey()
    cv2.destroyAllWindows()
# Use the keypoints to stitch the images
def stitch_image(img1, img2, M):
    #Get width and height of input images
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    #Get Compositve Dimension
    img1_dimension = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
    img2_temp_dimension = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)
    
    #Get relative perspective of 2nd img
    img2_dims = cv2.perspectiveTransform(img2_temp_dimension, M)

    #Resultant Dimensions
    resultant_dim = np.concatenate( (img1_dimension, img2_dims), axis = 0)
    
    #Get images together
    #1st Calc the dimension of match points
    [xmin, ymin] = np.int32(resultant_dim.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(resultant_dim.max(axis=0).ravel() + 0.5)

    #Create output array for transform
    transform_distance =[-xmin, -ymin] 
    transform_array = np.array([[1, 0, transform_distance[0]], [0, 1, transform_distance[1]], [0,0,1]])

    #Warp Images onto conposite
    result_img = cv2.warpPerspective(img2, transform_array.dot(M), (xmax-xmin, ymax-ymin))
    output = result_img[transform_distance[1]:w1+transform_distance[1], transform_distance[0]:h1+transform_distance[0]]
    #result_img[transform_distance[1]:w1+transform_distance[1], transform_distance[0]:h1+transform_distance[0]] = img1
    return (output)

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

# Equalize Histogram of Color Images
def equalize_histogram_color(img):
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
	img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
	return img

# Main function definition
def main():
    i = 0
    folder = ("C:\\Users\\SeanM\\Source\\Repos\\Stitcher_Method\\Match_")
    #folder = ("C:\\Users\\SeanM\\Source\\Repos\\Stitcher_Method\\Stitching_Folder\\Dense Reconstruction.nvm.cmvs\\00\\visualize")
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        print("Img: ",filename, "read.")
        #img = equalize_histogram_color(img)
        #print("Equalized")
        #img = cv2.resize(img,None,fx=.25, fy=.25)
        print("Resized")
        #img = pX.blackRect(img)
        print("Stitching img: ", filename)
        if img is not None:
                if i==0:
                    instance = img
                    i = i+1
                else: 
                    M =  get_sift_homography(equalize_histogram_color(instance), equalize_histogram_color(img))
                    instance = stitch_image(img, instance, M)

    cv2.imwrite('All.jpg', instance) 
    print("written to disk")

# Call main function
if __name__=='__main__':
	main()
