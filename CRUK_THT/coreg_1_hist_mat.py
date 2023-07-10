#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:24:10 2023

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

base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output'
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
hist_img_int[hist_img_int>230]=0

#%% Mask for circular ROI
hist_img_msk=hist_img_int
hist_img_msk[hist_img_msk>0]=255

hist_img_msk_inv=cv2.bitwise_not(hist_img_msk)

h, w = hist_img_int.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

hist_img_msk_fill=hist_img_msk

plt.figure(2)
plt.subplot(1,3,1)
plt.imshow(hist_img_msk_inv,cmap='gray')
plt.show()

cv2.floodFill(hist_img_msk_fill,mask,(h//2,w//2),255)

hist_img_msk_fill_inv=cv2.bitwise_not(hist_img_msk_fill)

plt.figure(2)
plt.subplot(1,3,2)
plt.imshow(hist_img_msk_fill,cmap='gray')
plt.show()

plt.figure(2)
plt.subplot(1,3,3)
plt.imshow(hist_img_msk_fill_inv,cmap='gray')
plt.show()
#%%




#%%
plt.figure(1)
plt.subplot(1,3,1)
plt.imshow(hist_img_gray,cmap='gray')
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(hist_img_val,cmap='gray')
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(hist_img_int,cmap='gray')
plt.colorbar()
plt.show()

#%%

