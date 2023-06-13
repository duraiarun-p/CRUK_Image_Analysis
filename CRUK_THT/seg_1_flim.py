#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:01:36 2023

@author: Arun PDRA, THT
"""
#%%


import cv2
from seg_est_lib import comp_tiling
from matplotlib import pyplot as plt


#%% Data loading and tiling
mypath='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-5_Col-11_20230224/FLT_IMG_DIR_4'
# mypath='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-9_20230222/FLT_IMG_DIR'
# mypath='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-4_Col-7_20230220/FLT_IMG_DIR'

# onlyfiles = [join(mypath, f) for f in listdir(mypath) if isdir(join(mypath, f))]



tma_int_f,cube_flt_f,cube_int_f=comp_tiling(mypath)

#%% 





#%% Data Visualisation
plt.figure(1),
plt.imshow(tma_int_f,cmap='gray')
plt.show()
