#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 09:46:43 2023

@author: arun
"""

import cv2

import numpy as np

import matplotlib.pyplot as plt

#%%

img=cv2.imread('/home/arun/Pictures/Test.jpg')

orow,ocol,och=img.shape



#%%
oP=[250,1000]

plt.figure(1)
plt.imshow(img)
plt.plot(oP[0],oP[1],marker='o',color='r')


#%%

# nrow,ncol=1024,512

# img_res=cv2.resize(img,(nrow,ncol),interpolation=cv2.INTER_NEAREST)
# nrow_rat,ncol_rat=nrow/orow,ncol/orow



# nP=oP
# nP[0]=oP[0]*nrow_rat
# nP[1]=oP[1]*ncol_rat

# plt.figure(2)
# plt.imshow(img_res)
# plt.plot(nP[0],nP[1],marker='o',color='r')

#%%
nrow,ncol=int(orow/2),int(ocol/2)

img_res=cv2.resize(img,(ncol,nrow),interpolation=cv2.INTER_NEAREST)
nrow_rat,ncol_rat=nrow/orow,ncol/orow

# # # input_pts = np.float32([[0,0], [ocol-1,0], [0,orow-1]])
# # # output_pts = np.float32([[ncol-1,0], [0,0], [ncol-1,nrow-1]])

# # input_pts = np.float32([[0,0], [(orow-1)/2,(ocol-1)/2], [orow-1,ocol-1]])
# # output_pts = np.float32([[0,0], [(nrow-1)/2,(ncol-1)/2], [nrow-1,ncol-1]])

# # M = cv2.getAffineTransform(input_pts , output_pts)

nP=oP
nP[0]=oP[0]*nrow_rat
nP[1]=oP[1]*ncol_rat

# # nP=np.array(oP)@M

plt.figure(3)
plt.imshow(img_res)
plt.plot(nP[0],nP[1],marker='o',color='r')

#%%
def Affine_OpCV_2D(Fixed_sitk,Moving_sitk,number_of_iterations):
    
    def my_norm(x):
        x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
        return x_norm
    def prepare_img_4_reg_Fixed_changedatatype(Fixed,Moving):
        Fixed_datatype=str(Fixed.dtype)
        Moving_datatype=str(Moving.dtype)
        
        Fixed_N=np.zeros_like(Fixed)
        
        # Fixed=np.ascontiguousarray(Fixed,dtype=Moving_datatype)
        # Fixed_N=np.ascontiguousarray(Fixed_N,dtype=Moving_datatype)
    
        # Fixed_N = np.round(cv2.normalize(Fixed,  Fixed_N, 0, 255, cv2.NORM_MINMAX))
        # Fixed_N = cv2.normalize(Fixed,  Fixed_N, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        Fixed_N = my_norm(Fixed)*255 # for FLT intensity range to [0 255] image intensity range
        Fixed_N = Fixed_N.astype(Moving_datatype)
        
        Moving_N=np.zeros_like(Moving)
        Moving_N = np.round(cv2.normalize(Moving,  Moving_N, 0, 255, cv2.NORM_MINMAX))
        return Fixed_N, Moving_N

    Fixed_sitk, Moving_sitk=prepare_img_4_reg_Fixed_changedatatype(Fixed_sitk,Moving_sitk)
    
    sz=Fixed_sitk.shape
    # Moving_sitk=cv2.resize(Moving_sitk, (sz[1],sz[0]), interpolation= cv2.INTER_NEAREST)
    
    warp_mode = cv2.MOTION_AFFINE

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # number_of_iterations = 10000


    termination_eps = 1e-10

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    (cc, warp_matrix) = cv2.findTransformECC (Fixed_sitk,Moving_sitk,warp_matrix, warp_mode, criteria)

    # # warp_matrix=cv2.getAffine

    Moving_sitk_registered = cv2.warpAffine(Moving_sitk, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    return Moving_sitk_registered, warp_matrix, cc

#%%
# Fixed_sitk=img_res[:,:,1]
# Moving_sitk=img[:,:,1]
# number_of_iterations=1000
# Moving_sitk_registered, warp_matrix, cc=Affine_OpCV_2D(Fixed_sitk,Moving_sitk,number_of_iterations)
#%%
# img_res=np.float32(img_res)
# img=np.float32(img)
# warp_mode = cv2.MOTION_AFFINE
# warp_matrix = np.eye(2, 3, dtype=np.float32)
# number_of_iterations = 1000
# termination_eps = 1e-10
# criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
# (cc, warp_matrix) = cv2.findTransformECC (img_res,img,warp_matrix, warp_mode, criteria)
# # # warp_matrix=cv2.getAffine
# sz=img_res.shape
# img_re_or = cv2.warpAffine(img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#%%
# plt.figure(5)
# plt.imshow(Moving_sitk_registered,cmap='gray')
#%%
# warp_matrix_inv_1=np.linalg.inv(warp_matrix[:2,:2])
# warp_matrix_inv_2=np.zeros_like(warp_matrix_inv_1)
# warp_matrix_inv_2[:,0]=warp_matrix[:2,2]
# warp_matrix_inv_2[:,1]=warp_matrix[:2,2]
# warp_matrix_inv_2=np.linalg.inv(warp_matrix_inv_2)
# warp_matrix_inv_2=np.linalg.inv(np.array([warp_matrix[:2,2],warp_matrix[:2,2]]))
# nP=np.array(oP)@warp_matrix_inv

# plt.figure(6)
# plt.imshow(img_res)
# plt.plot(nP[0],nP[1],marker='o',color='r')
#%%
# old_corners=np.float32([[0,0],
#              [orow-1,0],
#              [orow-1,ocol-1],
#              [0,ocol-1]
#     ])

# new_corners=np.float32([[0,0],
#              [nrow-1,0],
#              [nrow-1,ncol-1],
#              [0,ocol-1]
#     ])

old_corners=np.float32([[0,0],
              [ocol-1,0],
              [ocol-1,orow-1],
              [0,orow-1]
    ])

new_corners=np.float32([[0,0],
              [nrow-1,0],
              [ncol-1,nrow-1],
              [0,orow-1]
    ])

# warp_matrix_perspective = cv2.getPerspectiveTransform(old_corners, new_corners)   
# new_corners_f=cv2.perspectiveTransform(old_corners,warp_matrix)[0]

# h, status = cv2.findHomography(old_corners, new_corners)

# #%%
# oP.append(1)
# # # nP=np.dot(warp_matrix_perspective,oP)
# # # warp_matrix[:2,:2]=warp_matrix_inv_1
# # # warp_matrix[:2,2]=warp_matrix_inv_2[:,0]
# # nP=np.dot(oP,warp_matrix)
# # # nP=np.dot(warp_matrix,oP)

# nP=np.dot(h,oP)
# plt.figure(6)
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
NFN=1000
aligned_img, homography, mask=OCV_Homography_2D(imgRef_grey,imgTest_grey,NFN)
#%%
plt.figure(10)
plt.imshow(aligned_img,cmap='gray')
#%%
oP1=[250,1000]
oP1.append(1)
# # # nP=np.dot(warp_matrix_perspective,oP)
# # # warp_matrix[:2,:2]=warp_matrix_inv_1
# # # warp_matrix[:2,2]=warp_matrix_inv_2[:,0]
# # nP=np.dot(oP,warp_matrix)
# # # nP=np.dot(warp_matrix,oP)

# nP=np.dot(homography,oP)
# oP1=oP
oP1=np.array(oP1,dtype=np.int32)
oP1[0]=oP1[0]/orow
# oP1[1]=oP1[1]/ocol
# oP1[2]=oP1[2]/1
nP=homography@oP
# nPsum = np.sum(nP ,1)
# nP=cv2.perspectiveTransform(oP,homography)

# new_corners_f=cv2.perspectiveTransform(old_corners,homography)[0]

# oP_a=np.array(oP).reshape((1,2))
# oP_h=cv2.convertPointsToHomogeneous(oP_a)
# nP_h=np.dot(homography,oP_h)
# nP_a=cv2.convertPointsFromHomogeneous(nP_h)

# plt.figure(11)
# plt.imshow(img_res)
# plt.plot(nP[0],nP[1],marker='o',color='r')