#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:17:43 2023

@author: Arun PDRA, THT
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
import resource
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

import os
from os import listdir
from os.path import join, isdir
from timeit import default_timer as timer
from scipy import ndimage as ndi
import imageio
from xtiff import to_tiff

import cv2
#%%

mypath='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-5_Col-11_20230224/FLT_IMG_DIR_4'
# mypath='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-9_20230222/FLT_IMG_DIR'

# onlyfiles = [join(mypath, f) for f in listdir(mypath) if isdir(join(mypath, f))]
onlyfiles = [join(mypath, f) for f in listdir(mypath)]
onlyfiles.sort()
#del onlyfiles[-1]# remove cube file from list
onlyfiles_len=len(onlyfiles)

flt_cube_list=[]
int_cube_list=[]
int_list=[]

for tile_file in range(onlyfiles_len):
    mat_fname=onlyfiles[tile_file]
    print(mat_fname)
    mat_contents = sio.loadmat(mat_fname)
    img_flt_ref=mat_contents['img_flt']
    img_int_ref=mat_contents['img_int']
    img_flt=img_flt_ref[()]
    img_int=img_int_ref[()]
    img_flt=np.moveaxis(img_flt, -1, 0)
    img_int=np.moveaxis(img_int, -1, 0)
    
    # filename=str(tile_file)+'.tiff'
    filename='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/Test_Output/Tiling/Tile_'+str(tile_file+1)+'.tif'
    # Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/Test_Output/Tiling
    img_int_sl=np.sum(img_int[:,:,:],axis=0)
    imageio.imwrite(filename,img_int_sl)
    
    flt_cube_list.append(img_flt)
    int_cube_list.append(img_int)
    int_list.append(img_int_sl)
    
# imageio.imwrite('myimgs.tiff',img_int)

# to_tiff(img_int,'test.ome.tiff')
#%%
# a=np.array([1,4,7,2,5,8,3,6,9])
# ai=np.argsort(a)
# # ai_1=list(ai)
# # int_list.sort(key=ai)
# # int_list = [int_list[i] for i in ai]
# def resort_list(int_list1):
#     temp=int_list1
#     for i in range(len(int_list1)):
#         temp[ai[i]]=int_list1[i]
#         # print(ai[i])
#     return temp

# int_list1=resort_list(int_list)






#%%
tile_1=int_list[0]
tile_2=int_list[1]
tile_3=int_list[2]

tile_1=int_list[3]
tile_2=int_list[4]
tile_3=int_list[5]

tile_1=int_list[6]
tile_2=int_list[7]
tile_3=int_list[8]


# stitchy=cv2.Stitcher.create()
# (dummy,output)=stitchy.stitch(int_list[0:2])

#%%
plt.figure(1),
plt.subplot(1,3,1)
plt.imshow(tile_1,cmap='gray')
plt.subplot(1,3,2)
plt.imshow(tile_2,cmap='gray')
plt.subplot(1,3,3)
plt.imshow(tile_3,cmap='gray')
plt.show()



# import stitching

# stitcher = stitching.Stitcher()
# panorama = stitcher.stitch([tile_1,tile_2])
#%%

def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    
    list_of_points_1 = np.float32([
        [0,0], 
        [0,rows1],
        [cols1,rows1], 
        [cols1,0]
    ])
    list_of_points_1 = list_of_points_1.reshape(-1,1,2)

    temp_points = np.float32([
        [0,0], 
        [0,rows2], 
        [cols2,rows2],
        [cols2,0]
    ])
    temp_points = temp_points.reshape(-1,1,2)
    
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)
    
    ##Define boundaries:
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    
    translation_dist = [-x_min,-y_min]
    
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])
    
    output_img = cv2.warpPerspective(img2, 
                                     H_translation.dot(H), 
                                     (x_max - x_min, y_max - y_min))
    ## Paste the image:
    output_img[translation_dist[1]:rows1+translation_dist[1], 
               translation_dist[0]:cols1+translation_dist[0]] = img1
    
    return output_img

def view_stitch(img_1,img_2):
    img1 = cv2.normalize(img_1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    img2 = cv2.normalize(img_2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    cv2.imshow('img1',img1)
    
    sift = cv2.SIFT_create()
    # sift = cv2.ORB_create()
    # find the key points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    #cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(img_,kp1,None))

    match_ob = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    # match = cv2.BFMatcher()
    # matches = match.knnMatch(des1,des2,k=2)
    
    matches = match_ob.match(des1, des2)

    # good = []
    # for m,n in matches:
    #     if m.distance < 0.95*n.distance:
    #         good.append(m)
            
    # if len(good) > 10:
    #     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    good = matches
    # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
    src_pts = np.float32([kp1[i.queryIdx].pt for i in matches])
    dst_pts = np.float32([kp2[i.trainIdx].pt for i in matches])

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
    # else:
    #     M = 0

    
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    cv2.imshow("original_image_drawMatches", img3)
    
    
    warped_image = cv2.warpPerspective(img_1,M,(img_1.shape[0]*2, img_1.shape[1]*2))
    # warped_image=warpImages(img1, img2, M)
    
    cv2.imshow('stitched',warped_image)
    
    # return M
    
    
#%%
# M=view_stitch(tile_1, tile_2)
# tile_1 = cv2.normalize(tile_1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
# tile_2 = cv2.normalize(tile_2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

# res = cv2.matchTemplate(tile_1, tile_2, cv2.TM_CCOEFF_NORMED)
  
# # Specify a threshold
# threshold = 0.8
  
# # Store the coordinates of matched area in a numpy array
# loc = np.where(res >= threshold)
  
# # Draw a rectangle around the matched region.
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(tile_1, pt, (pt[0] + 100, pt[1] + 100), (0, 255, 255), 2)
    
# cv2.imshow('Detected', tile_1)