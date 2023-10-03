#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 22:30:05 2023

@author: Arun PDRA, THT
"""

import h5py
# from scipy.io import loadmat

#%%


# matfile_list_path='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-2_Col-3_20230216/Row_1_Col_1/workspace.frame_1.mat'

matfile_list_path='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output/1.mat'

mat_contents=h5py.File(matfile_list_path,'r+')
mat_contents_list=list(mat_contents.keys())

# mat_contents = loadmat(matfile_list_path)
# mat_contents_list=list(mat_contents.keys())

# bin_array_ref=mat_contents['bins_array_3']
# frame_size_x_ref=mat_contents['frame_size_x']
# hist_mode_ref=mat_contents['HIST_MODE']
# binWidth_ref=mat_contents['binWidth']

# bin_array0=bin_array_ref[()]
# frame_size=int(frame_size_x_ref[()])
# hist_mode=int(hist_mode_ref[()])
# binWidth=float(binWidth_ref[()])# time in ns


allIntensityImages1_ref=mat_contents['allIntensityImages1']
allIntensityImages1=allIntensityImages1_ref[()]

lifetimeAlphaData1_ref=mat_contents['lifetimeAlphaData1']
lifetimeAlphaData1=lifetimeAlphaData1_ref[()]

lifetimeImageData1_ref=mat_contents['lifetimeImageData1']
lifetimeImageData1=lifetimeImageData1_ref[()]