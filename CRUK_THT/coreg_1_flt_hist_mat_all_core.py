#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 13:45:18 2023

@author: Arun PDRA, THT
"""

#%%
import os
# import sys
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# import h5py
# from scipy import ndimage as ndi
# from scipy.io import savemat,loadmat
# from coreg_lib import coreg_img_pre_process,OCV_Homography_2D,prepare_img_4_reg_Moving_changedatatype,prepare_img_4_reg_Fixed_changedatatype,Affine_OpCV_2D,perf_reg,warp_flt_img_3D
# from coreg_lib import *
import coreg_lib as cr
# import time
#%%

base_dir_lis=[]
base_dir_lis.append('/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output')
base_dir_lis.append('/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output')
base_dir_lis.append('/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output')
base_dir_lis.append('/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218/Mat_output2')
base_dir_lis.append('/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-4_Col-1_20230214/Mat_output2')
base_dir_lis.append('/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-6_Col-10_20230223/Mat_output2')

for base_dir in base_dir_lis:
    print(base_dir)
    cr.coreg_hist_int_flt(base_dir)
    print(f"{base_dir} is completed")