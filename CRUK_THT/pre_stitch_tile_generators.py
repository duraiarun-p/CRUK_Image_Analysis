#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 07:31:32 2023

@author: Arun PDRA, THT
"""

#%%
import os
from os import listdir
from os.path import join, isdir
import numpy as np
import cv2
import matplotlib.pyplot as plt

import h5py
from scipy import ndimage as ndi
from scipy.io import loadmat, savemat
import imageio

#%%

# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-4_Col-1_20230214/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-6_Col-10_20230223/Mat_output'


base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-9_20230222'




base_dir_splitted = base_dir.replace('/',',')
base_dir_splitted = base_dir_splitted.split(',')
tissue_core_file_name=base_dir_splitted[-1]
# save_path ='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/Test_Output'+'/'+tissue_core_file_name

base_dir_flt=base_dir+'/FLT_IMG_DIR/'
base_dir_flt_stitch=base_dir_flt+'/Stitched/'
if not os.path.exists(base_dir_flt_stitch):
    os.makedirs(base_dir_flt_stitch)

onlyfiles = [join(base_dir_flt, f) for f in listdir(base_dir_flt)]
onlyfiles.sort()
del onlyfiles[-1]# remove stictch file from list
onlyfiles_len=len(onlyfiles)

flt_cube_list=[]
int_cube_list=[]
int_list=[]


for tile_file in range(onlyfiles_len):
    mat_fname=onlyfiles[tile_file]
    print(mat_fname)
    mat_contents = loadmat(mat_fname)
    img_flt_ref=mat_contents['img_flt']
    img_int_ref=mat_contents['img_int']
    img_flt=img_flt_ref[()]
    img_int=img_int_ref[()]
    img_flt[img_flt>10]=10
    
    img_int_sl=np.sum(img_int,axis=-1)
    
    filename=base_dir_flt_stitch+str(tile_file+1)+'.png'
    # Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/Test_Output/Tiling
    img_int_sl=np.sum(img_int[:,:,:],axis=-1)
    imageio.imwrite(filename,img_int_sl)
