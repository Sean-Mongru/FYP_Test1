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
import operator
from PIL import Image
from PIL import ImageDraw

def show_img(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()



def alphaBlend(img1, img2, mask):
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if mask.ndim==3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended


def alpha_blend(img1, img2, mask):
    # Read the images
    foreground = img1
    background = img2
    alpha = mask
 
    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)
 
    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255
 
    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)
 
    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)
 
    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)
 
    # Display image
    cv2.imshow("outImg", outImage/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return outImage

def alphablend2(img1, img2):

    alpha = 0.5
    # [load]
    src1 = img1
    src2 = img2
    # [load]
    # [blend_images]
    beta = (1.0 - alpha)
    #dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
    dst = np.uint8(alpha*(img1)+beta*(img2))

    # [blend_images]
    # [display]
    #cv2.imshow('dst', dst)
    #cv2.waitKey(0)
    # [display]
    #cv2.destroyAllWindows()
    return dst




def stitch_image(img1, img2, M):
    #Get width and height of input images
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    #Get Compositve Dimension
    img1_dimension = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
    img2_temp_dimension = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)

    #Get relative perspective of 2nd img
    img2_dimension = cv2.perspectiveTransform(img2_temp_dimension, M)


    #Resultant Dimensions
    resultant_dim = np.concatenate( (img1_dimension, img2_dimension), axis = 0)

    #Get images together
    #1st Calc the dimension of match points
    [xmin, ymin] = np.int32(resultant_dim.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(resultant_dim.max(axis=0).ravel() + 0.5)

    #Create output array for transform
    transform_distance =[-xmin, -ymin] 
    transform_array = np.array([[1, 0, transform_distance[0]], [0, 1, transform_distance[1]], [0,0,1]])

    #Warp Image onto conposite
    result_img = cv2.warpPerspective(img2, transform_array.dot(M), (xmax-xmin, ymax-ymin))


    #show_img("Img2 + Composite",result_img)#########
    (t, maskwith_img2) = cv2.threshold(result_img, 1, 255, cv2.THRESH_BINARY)#######Image 2 Masked

    #show_img("Img2 Mask",maskwith_img2)#############
    ##Masking
    height,width,depth = result_img.shape
    mask_1 = np.zeros((height,width))
    mask_1 = cv2.merge([mask_1, mask_1, mask_1])

    foreground = mask_1

    foreground = foreground.astype(np.uint8)

    foreground[transform_distance[1]:w1+transform_distance[1], transform_distance[0]:h1+transform_distance[0]] = img1
    print (foreground.dtype)
    print (img1.dtype)

    #show_img("Img1+Composite", foreground)##########


    # blur and grayscale before thresholding
    blur = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)

    # perform inverse binary thresholding 
    (t, maskLayer) = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)

    # make a mask suitable for color images
    mask_2 = cv2.merge([maskLayer, maskLayer, maskLayer])

    mask_1[transform_distance[1]:w1+transform_distance[1], transform_distance[0]:h1+transform_distance[0]] = mask_2 #Image 1 mask
    #show_img("img1 mask", mask_1)

    
    
    ret,foreground_mask = cv2.threshold(mask_1,1,255,cv2.THRESH_BINARY)
    ret,background_mask = cv2.threshold(maskwith_img2,1,255,cv2.THRESH_BINARY)
    foreground_mask = foreground_mask.astype(np.uint8)
    print (foreground_mask.dtype)
    print (background_mask.dtype)



    inside_mask = foreground_mask & background_mask
    #show_img("inside_mask", inside_mask) ###########


    #inside_mask = get_distanceTransform(inside_mask)
    #show_img("inside_mask", inside_mask) 

    #result_img = alpha_blend(foreground, result_img, maskwith_img2)
    
    #result_img = alphablend2(foreground, result_img)

    result_img[transform_distance[1]:w1+transform_distance[1], transform_distance[0]:h1+transform_distance[0]] = img1


        #####Testing
    result_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)



    _, thresh = cv2.threshold(result_gray, 1, 255, cv2.THRESH_BINARY)
    
    dino, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
    print ("Found %d contours..." % (len(contours)))

    max_area = 0
    
    best_rect = (0,0,0,0)

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        deltaHeight = h-y

        deltaWidth = w-x	

        area = deltaHeight * deltaWidth	

        if ( area > max_area and deltaHeight > 0 and deltaWidth > 0):
            max_area = area
            best_rect = (x,y,w,h)
        if ( max_area > 0 ):
            final_img_crop = result_img[best_rect[1]:best_rect[1]+best_rect[3],
                best_rect[0]:best_rect[0]+best_rect[2]]
            result_img = final_img_crop

        #small = cv2.resize(final_img, (0,0), fx=0.15, fy=0.15) 

            return result_img


        else:
            print("Dud")




    #show_img("result_img", result_img)###############
   # return (result_img)