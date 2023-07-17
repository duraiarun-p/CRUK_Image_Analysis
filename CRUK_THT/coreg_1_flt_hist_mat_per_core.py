#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 12:09:30 2023

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
# from coreg_lib import coreg_img_pre_process,OCV_Homography_2D,prepare_img_4_reg_Moving_changedatatype,prepare_img_4_reg_Fixed_changedatatype,Affine_OpCV_2D,perf_reg,warp_flt_img_3D
# from coreg_lib import *
import coreg_lib as cr
import time
#%%
#%%

base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output'
base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output'
base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output'
base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218/Mat_output2'
base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-4_Col-1_20230214/Mat_output2'
base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-6_Col-10_20230223/Mat_output2'

#%% Hist Image Loading with assertion
#Loading Hist image automatically
file_extension_type = ('.tif',) # , '.exe', 'jpg', '...')
for hist_file in os.listdir(base_dir):
    if hist_file.endswith(file_extension_type) and hist_file.startswith('R'):
        print("Found a file {}".format(hist_file)) 
        hist_img=cv2.imread(f"{base_dir}/{hist_file}")
        
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
hist_img_hsv_f,hist_img_f,hist_img_gray_f,hist_mask,hist_img_gray=cr.coreg_img_pre_process(hist_img,thresh)

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


#%% Saturation - Hist Registration
Fixed=stitch_intensity
Moving=hist_img_hsv_f[:,:,1]
NofFeaturs=1000
NofIterations=10000

Fixed_N, Moving_N=cr.prepare_img_4_reg_Fixed_changedatatype(Fixed,Moving)

tic = time.perf_counter()
Moving_R3, warp_matrix, cc=cr.Affine_OpCV_2D(Fixed_N,Moving_N,NofIterations)
toc = time.perf_counter()
Affine_time=(toc-tic)/60
print('Affine: %s'%Affine_time)

#%% Registration Evaluation

Reg_SH=cr.perf_reg(Fixed_N,Moving_R3)

#%% Visualisation

plt.figure(6)
plt.subplot(1,3,1)
plt.imshow(Fixed_N,cmap='gray')
plt.colorbar()
plt.title('Fixed')
plt.subplot(1,3,2)
plt.imshow(Moving_N,cmap='gray')
plt.colorbar()
plt.title('Moving - S ch')
plt.subplot(1,3,3)
plt.imshow(Moving_R3,cmap='gray')
plt.colorbar()
plt.title('Moving registered affine')
plt.show()

#%% Co-registration for the whole core - hyperspectral image cube

# Need to apply mask for 3D
sz_fixed=Fixed_N.shape
Moving_sitk_int=stitch_intensity_cube
Moving_sitk_registered_int=stitch_intensity_cube
# Moving_sitk_registered_int=cr.warp_flt_img_3D(warp_matrix,sz_fixed,Moving_sitk_int)
Moving_sitk_flt=stitch_flt_cube
Moving_sitk_registered_flt=stitch_flt_cube
# Moving_sitk_registered_flt=cr.warp_flt_img_3D(warp_matrix,sz_fixed,Moving_sitk_flt)


#%%
Moving_sitk=hist_img_f
Moving_R4=cr.warp_flt_img_3D(warp_matrix,sz_fixed,Moving_sitk)
cv2.imwrite(f"{base_dir}/hist_registered.tiff", Moving_R4)




#%%
page=50
plt.figure(7)
# plt.subplot(2,2,1)
# plt.imshow(Moving_sitk_int[:,:,page],cmap='gray')
# plt.colorbar()
# plt.subplot(2,2,2)
# plt.imshow(Moving_sitk_registered_int[:,:,page],cmap='gray')
# plt.colorbar()
# plt.subplot(2,2,3)
# plt.imshow(Moving_sitk_flt[:,:,page],cmap='gray')
# plt.colorbar()
# plt.subplot(2,2,4)
# plt.imshow(Moving_sitk_registered_flt[:,:,page],cmap='gray')
# plt.colorbar()
plt.subplot(1,2,1)
plt.imshow(hist_img)
plt.subplot(1,2,2)
plt.imshow(Moving_R4)