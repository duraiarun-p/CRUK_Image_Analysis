#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:11:20 2023

@author: CRUK EDD - DAPS(PDRA)
"""
#% Importing Libraries
#%% Import Libraries
import pdb
import time
import datetime

import sys
import os
from os import listdir
from os.path import isfile, join, isdir

from matplotlib import pyplot as plt
from scipy.io import savemat
import h5py

import numpy as np
import numpy.ma as ma
import random
from scipy.optimize import curve_fit
from scipy import stats
from scipy import ndimage as ndi
from scipy import signal as sig
# from PIL import Image, ImageFilter
# from skimage.morphology import area_closing,flood
from skimage import filters,util
from skimage.filters import threshold_otsu

# from multiprocessing import Process
# from multiprocessing import Pool
import multiprocessing
# from numba import jit
# from numba import cuda

import concurrent.futures
from traceback import print_exc

from timeit import default_timer as timer 






n_cores = multiprocessing.cpu_count()

#%% Data Loading 


# "Windows way to pass path as an argument"
# mypath = str(r"C:\Users\CRUK EDD\Documents\Python_Scripts\Test_Data\Normal\Row-1_Col-1_20230303").replace("\\", "\\\\")
# mypath = str(r"C:\Users\CRUK EDD\Documents\Python_Scripts\Test_Data\Tumour\Row-1_Col-1_20230214").replace("\\", "\\\\")

def bin_array_loader(mypath,tile_file,time_index,spectral_index,spectral_span_sum):
    
# List of responses of all tiles
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isdir(join(mypath, f))]
    onlyfiles.sort()
    # Mat file per tile
    # tile_file=3
    matfile_list=listdir(onlyfiles[tile_file])
    #iterable tile
    matfile_list_path=join(onlyfiles[tile_file],matfile_list[0])#picking the mat file
    
    
    mat_contents=h5py.File(matfile_list_path,'r+')
    mat_contents_list=list(mat_contents.keys())
    
    bin_array_ref=mat_contents['bins_array_3']
    frame_size_x_ref=mat_contents['frame_size_x']
    hist_mode_ref=mat_contents['HIST_MODE']
    binWidth_ref=mat_contents['binWidth']
    
    bin_array0=bin_array_ref[()]
    frame_size=int(frame_size_x_ref[()])
    hist_mode=int(hist_mode_ref[()])
    binWidth=float(binWidth_ref[()])# time in ns
    
    ##%% Array Slicing based on spectrum/wavelength and parameter selection
    time_interval=binWidth
    
    time_resolution=(binWidth*1000)/(2*2**hist_mode)# time unit in ps
    
    bin_size=np.shape(bin_array0)
    # time_index=2
    
    time_indices=np.arange(bin_size[time_index])
    time_line=time_indices*time_interval# Time axis for fitting data
    
    # spectral_index=100
    # spectral_span_sum=32
    # spectral_span_sum=16
    bin_size=np.shape(bin_array0)
    
    # bin_mean=np.mean(bin_array0)
    
    #Similar to movsum
    # bin_array=np.sum(bin_array0[:,:,:,spectral_index],-1)
    bin_array=np.sum(bin_array0[:,:,:,spectral_index:spectral_index+spectral_span_sum],-1)
    # bin_array_bck=np.mean(bin_array,2)# Selecting background based on time bin
    # bin_mean=np.mean(bin_array)
    
    # Background subtraction
    # for time_bin in range(bin_size[time_index]):    
    #     bin_array[:,:,time_bin]=bin_array[:,:,time_bin]-bin_array_bck
    
    ##%% An attempt to parallelize the loops
    
    # def dilate(cycles, image):
    #    image = Image.fromarray(image, 'L')
    #    for _ in range(cycles):
    #        image = image.filter(ImageFilter.MaxFilter(3))
    #    return np.array(image)
    
    # bin_int=bin_array0[:,:,:,spectral_index]# only for spectral span = 0
    
    bin_int=bin_array[:,:,:]
    
    # bin_int_array_4_mask=bin_array[:,:,0]
    
    bin_int_array=np.cumsum(bin_int,axis=time_index)
    bin_int_array=bin_int_array[:,:,-1]
    
    
    bin_resp_spec_0=np.zeros((bin_size[-1],1))
    for spec_i in range(bin_size[-1]):
        # bin_resp_spec[spec_i]=np.sum(bin_array0[:,:,:,spec_i])
        bin_resp_spec_0[spec_i]=np.sum(bin_array0[:,:,15,spec_i])
    
    spec_indices=np.linspace(start = 1, stop = bin_size[-1], num = bin_size[-1])
    
    # spectral_indices=np.arange(0,350,8)
    # bin_resp_spec_01=bin_resp_spec_0[spectral_indices]
    # spec_indices1=spec_indices[spectral_indices]
    
    return bin_array,bin_int_array,bin_resp_spec_0,spec_indices

#%%
tile_file=3
time_index=2
spectral_index=100
spectral_span_sum=32
"Windows way to pass path as an argument"
mypath_N = str(r"C:\Users\CRUK EDD\Documents\Python_Scripts\Test_Data\Normal\Row-1_Col-1_20230303").replace("\\", "\\\\")
bin_array_N,bin_int_array_N,bin_spec_N,spec_indices=bin_array_loader(mypath_N,tile_file,time_index,spectral_index,spectral_span_sum)
# mypath_NB = str(r"C:\Users\CRUK EDD\Documents\Python_Scripts\Test_Data\Normal\background_20230308").replace("\\", "\\\\")
mypath_NB = str(r"C:\Users\CRUK EDD\Documents\Python_Scripts\Test_Data\Tumour\Row-1_Col-1_20230214").replace("\\", "\\\\")
bin_array_NB,bin_int_array_NB,bin_spec_NB,spec_indices=bin_array_loader(mypath_NB,tile_file,time_index,spectral_index,spectral_span_sum)

#%%
plt.figure(14)
plt.imshow(bin_int_array_NB,cmap='gray')
plt.colorbar()
plt.show()
plt.title('Tumour Intensity')

plt.figure(5)
plt.imshow(bin_int_array_N,cmap='gray')
plt.colorbar()
plt.show()
plt.title('Normal Intensity ')
#%%

spectrum_step=32
spectral_indices=np.arange(0,350,spectrum_step)

spec_indices_sl=spec_indices[spectral_indices]
bin_spec_N_sl=bin_spec_N[spectral_indices]
bin_spec_NB_sl=bin_spec_NB[spectral_indices]

plt.figure(502)
    # x=time_bin_indices_selected
    # y=bin_resp_selected
ax = plt.axes()
ax.scatter(spec_indices_sl,bin_spec_N_sl, c='blue', marker='o', edgecolors='b', s=15, label='Normal - chosen')
ax.scatter(spec_indices_sl,bin_spec_NB_sl, c='red', marker='o', edgecolors='r', s=15, label='Tumour - chosen')

ax.plot(spec_indices,bin_spec_N,'c', label='Normal')
ax.plot(spec_indices,bin_spec_NB,'g', label='Tumour')

# ax.plot(spec_indices_sl,bin_spec_N_sl,'c', label='Normal')
# ax.plot(spec_indices_sl,bin_spec_NB_sl,'g', label='Tumour')
ax.set_title(r'Fluorecence spectrum')
ax.set_xlabel(r'Spectrum Index')
ax.set_ylabel(r'Intensity (counts)')
ax.set_xlim(-0.1)
ax.set_ylim(-0.1)
ax.legend(fontsize=8)
plt.show()
#%% Choosing specific wavelength

