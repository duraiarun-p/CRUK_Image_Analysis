#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:47:28 2023

@author: arun
"""
import cv2

import numpy as np

import matplotlib.pyplot as plt
#%%

# img=cv2.imread('/home/arun/Pictures/Test.jpg')
img=cv2.imread('/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/Test.jpg')

ocol,orow,och=img.shape



#%%
oP=np.array([400,400])
oP1=[400,400]

plt.figure(1)
plt.imshow(img)
plt.plot(oP[0],oP[1],marker='o',color='r')

#%%
# nrow,ncol=int(orow/2),int(ocol/2)

nrow,ncol=2000,1000

img_res=cv2.resize(img,(ncol,nrow),interpolation=cv2.INTER_NEAREST)
nrow_rat,ncol_rat=nrow/orow,ncol/ocol

# oP[0]=oP[0]-(orow/2-1)
# oP[1]=oP[0]-(ocol/2-1)

nP=np.zeros_like(oP)
nP[0]=oP[0]*nrow_rat
nP[1]=oP[1]*ncol_rat

# nP[0]=nP[0]+(nrow/2-1)
# nP[1]=nP[0]+(ncol/2-1)

# plt.figure(3)
# plt.imshow(img_res)
# plt.plot(nP[0],nP[1],marker='o',color='r')

#%%
def OCV_Homography_2D(imgRef_grey,imgTest_grey,NFN):
    height, width = imgRef_grey.shape
    imgTest_grey=cv2.resize(imgTest_grey, (width,height), interpolation= cv2.INTER_NEAREST)
    orb_detector = cv2.ORB_create(nfeatures=NFN)
 
    # Extract key points and descriptors for both images
    keyPoint1, des1 = orb_detector.detectAndCompute(imgTest_grey, None)
    keyPoint2, des2 = orb_detector.detectAndCompute(imgRef_grey, None)
    
    # # Initiate SIFT detector
    # sift_detector = cv2.SIFT_create()
    # # Find the keypoints and descriptors with SIFT
    # keyPoint1, des1 = sift_detector.detectAndCompute(imgTest_grey, None)
    # keyPoint2, des2 = sift_detector.detectAndCompute(imgRef_grey, None)
    
    
    # Match features between two images using Brute Force matcher with Hamming distance
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matcher = cv2.BFMatcher()
    # Match the two sets of descriptors.
    matches = matcher.match(des1, des2)
     
    # # Sort matches on the basis of their Hamming distance.
    matches = sorted(matches, key=lambda x: x.distance)
    
     
    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 0.9)]
    
    # matches = matcher.knnMatch(des1, des2, k=2)

    # # Filter out poor matches
    # good_matches = []
    # for m,n in matches:
    #     if m.distance < 0.9*n.distance:
    #         good_matches.append(m)
    
    # matches = good_matches
    
    
    no_of_matches = len(matches)
    
        # Define 2x2 empty matrices
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
     
    # Storing values to the matrices
    for i in range(len(matches)):
        p1[i, :] = keyPoint1[matches[i].queryIdx].pt
        p2[i, :] = keyPoint2[matches[i].trainIdx].pt
     
    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    
    # Use homography matrix to transform the unaligned image wrt the reference image.
    aligned_img = cv2.warpPerspective(imgTest_grey, homography, (width, height))
    return aligned_img, homography, mask

imgRef_grey=img_res[:,:,1]
imgTest_grey=img[:,:,1]
NFN=5000
aligned_img, homography, mask=OCV_Homography_2D(imgRef_grey,imgTest_grey,NFN)
#%%
# plt.figure(10)
# plt.imshow(aligned_img,cmap='gray')
#%%
# oP1=[250,1000]

oP1.append(1)
# # # nP=np.dot(warp_matrix_perspective,oP)
# # # warp_matrix[:2,:2]=warp_matrix_inv_1
# # # warp_matrix[:2,2]=warp_matrix_inv_2[:,0]
# # nP=np.dot(oP,warp_matrix)
# # # nP=np.dot(warp_matrix,oP)

# nP=np.dot(homography,oP)
# oP1=oP
oP1=np.array(oP1,dtype=np.int32)
oP1=oP1/[orow,ocol,1]
# oP1x=np.zeros_like(oP1)
# oP1x[0]=oP1[0]/orow
# oP1x[1]=oP1[1]/ocol
# oP1x[2]=oP1[2]/1
nP1=homography@oP1
nP2=np.dot(oP1,homography)
nP2a=nP2*[nrow,ncol,1]
#%%
plt.figure(11)
plt.imshow(img_res)
plt.plot(nP2a[1],nP2a[0],marker='o',color='r')
#%%
nP=np.zeros_like(oP)

mid_r=nrow/2
mid_c=ncol/2




# nP[1] = mid_r
# nP[0] = mid_c

nP[0] = mid_r+ (nrow_rat*(oP[0]-mid_r))
nP[1] = mid_c+ (ncol_rat*(oP[1]-mid_c))

plt.figure(12)
plt.imshow(img_res)
plt.plot(nP[0],nP[1],marker='o',color='r')
plt.plot(mid_c,mid_r,marker='o',color='g')