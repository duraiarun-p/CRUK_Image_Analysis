#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:01:36 2023

@author: Arun PDRA, THT
"""
#%%


import cv2
from seg_est_lib import comp_tiling,mosaic_masking
from matplotlib import pyplot as plt

import numpy as np


#%% Data loading and tiling
mypath='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-5_Col-11_20230224/FLT_IMG_DIR_4'
mypath='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-9_20230222/FLT_IMG_DIR'
# mypath='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-4_Col-7_20230220/FLT_IMG_DIR'

# onlyfiles = [join(mypath, f) for f in listdir(mypath) if isdir(join(mypath, f))]



tma_int_f,cube_flt_f,cube_int_f=comp_tiling(mypath)

#%% 
blur_kernel_width=100
mask_kernel_width=50
#%%
tma_int_f_1,cube_flt_f_1,cube_int_f_1=mosaic_masking(tma_int_f,cube_flt_f,cube_int_f,blur_kernel_width,mask_kernel_width)


#%% Data Visualisation
# plt.figure(1)
# plt.subplot(1,2,1)
# plt.imshow(tma_int_f,cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(img,cmap='gray')
# plt.show()

# plt.figure(2)
# plt.subplot(1,2,1)
# plt.imshow(mask,cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(opening)
# plt.show()
#%%


#%%
page=7
plt.figure(3)
plt.subplot(1,3,1)
plt.imshow(tma_int_f_1,cmap='gray')
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(cube_int_f_1[:,:,page],cmap='gray')
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(cube_flt_f_1[:,:,page],cmap='gray')
plt.colorbar()