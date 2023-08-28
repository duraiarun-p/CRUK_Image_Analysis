#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:06:24 2023

@author: Arun PDRA, THT
"""



from scipy.io import savemat,loadmat
import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py

#%%
#%%

# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output'
base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218/Mat_output2'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-4_Col-1_20230214/Mat_output2'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-6_Col-10_20230223/Mat_output2'




#%%
core_mat_cont_file=base_dir+'/core_stitched_T.mat'
core_mat_contents=h5py.File(core_mat_cont_file,'r')
# core_mat_contents=loadmat(core_mat_cont_file)
core_mat_contents_list=list(core_mat_contents.keys())

stitch_intensity_ref=core_mat_contents['stitch_intensity']
stitch_intensity=stitch_intensity_ref[()]

stitch_intensity_cube_ref=core_mat_contents['stitch_intensity_cube']
stitch_intensity_cube=stitch_intensity_cube_ref[()]

stitch_flt_cube_ref=core_mat_contents['stitch_flt_cube']
stitch_flt_cube=stitch_flt_cube_ref[()]
#%%
thresh=10000
core_mask=stitch_intensity
core_mask = cv2.GaussianBlur(core_mask, (35,35),100)
core_mask[core_mask<thresh]=0
core_mask_edge = cv2.Sobel(src=core_mask, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=25)
core_mask_edge[core_mask_edge!=0]=255



kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(55,55))
core_mask_edge = cv2.morphologyEx(core_mask_edge,cv2.MORPH_OPEN,kernel)

core_mask_edge_inv=cv2.bitwise_not(core_mask_edge)
core_mask_edge_inv=np.nan_to_num(core_mask_edge_inv,copy=True, nan=0.0, posinf=None, neginf=None)
core_mask_edge_inv[core_mask_edge_inv<0]=2550
core_mask_edge_inv[core_mask_edge_inv==0]=255
core_mask_edge_inv[core_mask_edge_inv==2550]=0

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
core_mask_edge_inv = cv2.morphologyEx(core_mask_edge_inv,cv2.MORPH_OPEN,kernel)

flt_mask=core_mask_edge_inv
flt_mask[flt_mask==0]=1000
flt_mask[flt_mask==255]=0
flt_mask[flt_mask==1000]=255
flt_mask=flt_mask.astype(str(stitch_intensity.dtype))
##%%




plt.figure(1)
plt.subplot(2,2,1)
plt.imshow(stitch_intensity,cmap='gray')
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(core_mask,cmap='gray')
plt.colorbar()
plt.subplot(2,2,3)
plt.imshow(core_mask_edge,cmap='gray')
plt.colorbar()
plt.subplot(2,2,4)
plt.imshow(flt_mask,cmap='gray')
plt.colorbar()
#%% Applying mask on the array
flt_mask[flt_mask==255]=1
stitch_intensity_masked=np.multiply(stitch_intensity,flt_mask)
flt_cube_shape=stitch_flt_cube.shape
stitch_flt_cube_masked=stitch_flt_cube
stitch_intensity_cube_masked=stitch_intensity_cube
for page in range(flt_cube_shape[2]):
    # stitch_flt_cube_masked[:,:,page]=np.multiply(stitch_flt_cube[:,:,page],flt_mask)
    # stitch_intensity_cube_masked[:,:,page]=np.multiply(stitch_intensity_cube[:,:,page],flt_mask)
    
    stitch_flt_cube_masked[page,:,:]=np.multiply(stitch_flt_cube[page,:,:],flt_mask)
    stitch_intensity_cube_masked[page,:,:]=np.multiply(stitch_intensity_cube[page,:,:],flt_mask)
    
# stitch_intensity_cube_masked[stitch_intensity_cube_masked>7000]=7000
# stitch_flt_cube_masked[stitch_intensity_cube_masked>7000]=10
#%%
page=100
plt.figure(2)
plt.subplot(1,3,1)
plt.imshow(stitch_intensity_masked,cmap='gray')
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(stitch_flt_cube_masked[:,:,page],cmap='gray')
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(stitch_intensity_cube_masked[:,:,page],cmap='gray')
plt.colorbar()

plt.figure(20)
plt.subplot(1,3,1)
plt.imshow(stitch_intensity,cmap='gray')
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(stitch_flt_cube[:,:,page],cmap='gray')
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(stitch_intensity_cube[:,:,page],cmap='gray')
plt.colorbar()
#%%
# stitch_flt=np.sum(stitch_flt_cube,axis=2)/flt_cube_shape[2]
# plt.figure(3)
# plt.imshow(stitch_flt,cmap='gray')
# plt.colorbar()
#%%
# thresh=0.5
# core_mask=stitch_flt
# core_mask = cv2.GaussianBlur(core_mask, (35,35),100)
# core_mask[core_mask<thresh]=0
# core_mask_edge = cv2.Sobel(src=core_mask, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=25)
# core_mask_edge[core_mask_edge!=0]=255

# plt.figure(4)
# plt.subplot(2,2,1)
# plt.imshow(stitch_flt,cmap='gray')
# plt.colorbar()
# plt.subplot(2,2,2)
# plt.imshow(core_mask,cmap='gray')
# plt.colorbar()
# plt.subplot(2,2,3)
# plt.imshow(core_mask_edge,cmap='gray')
# plt.colorbar()
# plt.subplot(2,2,4)
# plt.imshow(flt_mask,cmap='gray')
# plt.colorbar()
#%%
mdic={'stitch_intensity_masked':stitch_intensity_masked,'stitch_intensity_cube_masked':stitch_intensity_cube_masked,'stitch_flt_cube_masked':stitch_flt_cube_masked}
# savemat(f"{base_dir}/core_stitched_masked.mat", mdic)