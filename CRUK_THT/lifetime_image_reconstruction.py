#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:29:15 2023

@author: Arun PDRA, THT
"""

from os import listdir
from os.path import isfile, join, isdir

import h5py
import numpy as np
from timeit import default_timer as timer

from scipy import signal as sig
from skimage import filters

from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA

# import mat73

# from spectral import BandResampler

from scipy import ndimage as ndi

import multiprocessing

from lifetime_estimate_lib_THT import life_time_image_reconstruct_1_concurrent,prepare_data_list
#%%
def flt_img(bin_array2,spectral_index,time_index,fignum):
    
    bin_int=bin_array2[:,:,:,spectral_index]# only for spectral span = 1
    bin_array=bin_int
    bin_int_array=np.sum(bin_int,axis=time_index)
    
    bin_int_array1=filters.gaussian(bin_int_array, sigma=10)

    bin_mask_thresh=np.mean(bin_int_array1)*.75
    bin_int_array_mask=np.zeros_like(bin_int_array1)
    bin_int_array_mask[bin_int_array1>bin_mask_thresh]=1
        # bin_int_array_mask = filters.sobel(bin_int_array_mask)
        # bin_int_array_mask=area_closing(bin_int_array_mask,area_threshold=64)
    bin_int_array_mask = ndi.binary_fill_holes(bin_int_array_mask)
 
    plt.figure(fignum+1)
    plt.imshow(bin_int_array,cmap='gray')
    plt.colorbar()
    plt.show()
    plt.title('Intensity')
    # plt.figure(27)
    # plt.imshow(bin_int_array_mask,cmap='gray')
    # plt.colorbar()
    # plt.show()
    # plt.title('Intensity_mask')
    # plt.figure(28)
    # plt.imshow(bin_int_array_mask_edge,cmap='gray')
    # plt.colorbar()
    # plt.show()
    # plt.title('Intensity_mask')
    ##%%

    n_cores=multiprocessing.cpu_count()
    
    bin_Len,bin_list,time_list,bin_index_list=prepare_data_list(frame_size,bin_array,bin_size,time_index,time_indices,time_line)
    
    start_time_0=timer()
    # tau_1_array=life_time_image_reconstruct_1(frame_size,bin_Len,bin_list,time_line,bin_index_list,n_cores)  
    tau_1_array,r_1=life_time_image_reconstruct_1_concurrent(frame_size,bin_Len,bin_list,time_list,bin_index_list,n_cores)
    # tau_1_array=sig.medfilt2d(tau_1_array)
    # if (np.max(tau_1_array)-np.min(tau_1_array))/np.mean(tau_1_array)>2:
    #     tau_1_array[tau_1_array>np.mean(tau_1_array)+((np.max(tau_1_array)-np.min(tau_1_array))/20)]=0
    tau_1_array[tau_1_array>np.median(tau_1_array)*50]=0 # For visualisation
    # tau_1_array=sig.medfilt2d(tau_1_array)
    tau_1_array1=sig.medfilt2d(tau_1_array)
    # tau_1_array1 = ma.masked_array(tau_1_array, bin_int_array_mask)
    # tau_1_array1 = np.multiply(tau_1_array, bin_int_array_mask)
    # tau_1,r_1=life_time_image_reconstruct_1_gpu(frame_size,bin_Len,bin_list,time_line)
    runtimeN1=(timer()-start_time_0)/60
    print(runtimeN1)
    
    ##%%
    plt.figure(fignum)
    # plt.subplot(121)
    # plt.imshow(tau_1_array,cmap='gray')
    # plt.colorbar()
    # plt.subplot(122)
    plt.imshow(tau_1_array1,cmap='gray')
    plt.colorbar()
    plt.show()
    plt.title('Curvefit-Exp fitting with $R^2$:%.3f'%r_1)
    
    