#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 17:28:07 2023

@author: Arun PDRA, THT
"""

#%%
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

import h5py
from scipy import ndimage as ndi
from scipy.io import savemat
#%%

# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output'
base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218/Mat_output2'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-4_Col-1_20230214/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-6_Col-10_20230223/Mat_output'

#%% Hist Image Loading with assertion
#Loading Hist image automatically
file_extension_type = ('.tif',) # , '.exe', 'jpg', '...')
for hist_file in os.listdir(base_dir):
    if hist_file.endswith(file_extension_type) and hist_file.startswith('R'):
        print("Found a file {}".format(hist_file)) 
        hist_img=stitch_fiji=cv2.imread(f"{base_dir}/{hist_file}")
        
    # else:
        # print("File with the name was not found") 

if not 'hist_img' in locals():
    sys.exit("Execution was stopped due to Hist Image file was not found error")

#%% Coregistration

# Hist Image parameters
hist_img_shape=hist_img.shape
pix_x_hist=0.22
pix_x_hist=0.22

#%% Hist Image pre-processing for registration

hist_img_gray=cv2.cvtColor(hist_img, cv2.COLOR_BGR2GRAY)
hist_img_gray[hist_img_gray>200]=0

hist_img_hsv = cv2.cvtColor(hist_img, cv2.COLOR_BGR2HSV)
hist_img_val=hist_img_hsv[:,:,2]
hist_img_val[hist_img_val>200]=0

hist_img_int=cv2.bitwise_or(hist_img_gray,hist_img_val)
hist_img_int[hist_img_int>200]=0
#%% Mask for circular ROI
hist_img_msk=hist_img_int
hist_img_msk[hist_img_msk>0]=255

hist_img_msk_inv=cv2.bitwise_not(hist_img_msk)

h, w = hist_img_int.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

hist_img_msk_blur = cv2.GaussianBlur(hist_img_msk, (35,35),100)
hist_img_msk_edge = cv2.Sobel(src=hist_img_msk_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=25)

hist_img_msk_edge_con=hist_img_msk_edge
hist_img_msk_edge_con[hist_img_msk_edge_con==0]=255
hist_img_msk_edge_con[hist_img_msk_edge_con!=255]=0
hist_img_msk_edge_con=hist_img_msk_edge_con.astype('uint8')

hist_img_msk_edge_con_edge=cv2.Sobel(src=hist_img_msk_edge_con, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=25)
hist_img_msk_edge_con_edge[hist_img_msk_edge_con_edge==0]=255
hist_img_msk_edge_con_edge[hist_img_msk_edge_con_edge!=255]=0

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
hist_img_msk_edge_con_edge = cv2.morphologyEx(hist_img_msk_edge_con_edge,cv2.MORPH_OPEN,kernel)
hist_img_msk_edge_con_inv=cv2.bitwise_not(hist_img_msk_edge_con)



kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
hist_img_msk_edge_con_inv = cv2.morphologyEx(hist_img_msk_edge_con_inv,cv2.MORPH_OPEN,kernel)

contour,hier = cv2.findContours(hist_img_msk_edge_con_inv,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contour:
    cv2.drawContours(hist_img_msk_edge_con_inv,[cnt],0,255,-1)

gray = cv2.bitwise_not(hist_img_msk_edge_con_inv)

hist_mask= cv2.bitwise_not(gray)

hist_img_f=cv2.bitwise_and(hist_img_int,hist_mask)
#%%

plt.figure(1)
plt.imshow(hist_img_f,cmap='gray')
# plt.colorbar()
plt.show()