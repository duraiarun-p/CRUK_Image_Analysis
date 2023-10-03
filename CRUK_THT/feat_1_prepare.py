#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:18:14 2023

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
#%%
base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218/Mat_output2'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-4_Col-1_20230214/Mat_output2'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-6_Col-10_20230223/Mat_output2'
#%% Loa FLIM images
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
#%% Load ground truth labels with affine transformed coordinates