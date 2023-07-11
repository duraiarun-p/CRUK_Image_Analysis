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

# import h5py
from scipy import ndimage as ndi
from scipy.io import savemat,loadmat
from coreg_lib import coreg_img_pre_process,OCV_Homography_2D,prepare_img_4_reg_Moving_changedatatype,prepare_img_4_reg_Fixed_changedatatype
# import coreg_lib
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
thresh=200

#%% Mask applied
hist_img_hsv_f,hist_img_f,hist_img_gray_f,hist_mask,hist_img_gray=coreg_img_pre_process(hist_img,thresh)

#%% Stitched core mat file loading

core_mat_cont_file=base_dir+'/core_stitched.mat'
# core_mat_contents=h5py.File(core_mat_cont_file,'r+')
core_mat_contents=loadmat(core_mat_cont_file)
core_mat_contents_list=list(core_mat_contents.keys())

stitch_intensity_ref=core_mat_contents['stitch_intensity']
stitch_intensity=stitch_intensity_ref[()]

stitch_intensity_cube_ref=core_mat_contents['stitch_intensity_cube']
stitch_intensity_cube=stitch_intensity_cube_ref[()]

stitch_flt_cube_ref=core_mat_contents['stitch_flt_cube']
stitch_flt_cube=stitch_flt_cube_ref[()]

#%% Visulaisation
# plt.figure(1)
# plt.subplot(3,3,1)
# plt.imshow(hist_img)
# plt.subplot(3,3,2)
# plt.imshow(hist_img_gray,cmap='gray')
# plt.subplot(3,3,3)
# plt.imshow(hist_img_gray_f,cmap='gray')
# plt.subplot(3,3,4)
# plt.imshow(hist_img_f[:,:,0],cmap='gray')
# plt.subplot(3,3,5)
# plt.imshow(hist_img_f[:,:,1],cmap='gray')
# plt.subplot(3,3,6)
# plt.imshow(hist_img_f[:,:,2],cmap='gray')
# plt.subplot(3,3,7)
# plt.imshow(hist_img_hsv_f[:,:,0],cmap='gray')
# plt.subplot(3,3,8)
# plt.imshow(hist_img_hsv_f[:,:,1],cmap='gray')
# plt.subplot(3,3,9)
# plt.imshow(hist_img_hsv_f[:,:,2],cmap='gray')

# plt.figure(2)
# plt.subplot(2,2,1)
# plt.imshow(hist_img_gray_f,cmap='gray')
# plt.colorbar()
# plt.subplot(2,2,2)
# plt.imshow(stitch_intensity,cmap='gray')
# plt.colorbar()
# plt.show()
# plt.subplot(2,2,3)
# plt.imshow(hist_img_f[:,:,1],cmap='gray')
# plt.colorbar()
# plt.show()
# plt.subplot(2,2,4)
# plt.imshow(hist_img_hsv_f[:,:,2],cmap='gray')
# plt.colorbar()
# plt.show()
#%% Coregistration sub-routine

# Approach 1: Hist - Moving FLIM Intensity - Fixed
# Hist - Green Channel

Fixed=stitch_intensity
Moving=hist_img_f[:,:,1]
NofFeaturs=1000

Fixed_N, Moving_N=prepare_img_4_reg_Fixed_changedatatype(Fixed,Moving)
plt.figure(3)
plt.subplot(1,2,1)
plt.imshow(Fixed,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(Moving,cmap='gray')
plt.show()
#%%


Moving_R, homography, mask=OCV_Homography_2D(Fixed_N, Moving_N,NofFeaturs)


# Fixed_N, Moving_N=prepare_img_4_reg_Moving_changedatatype(Fixed,Moving)