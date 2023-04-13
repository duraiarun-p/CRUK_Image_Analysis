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

from lifetime_estimate_lib_THT import life_time_image_reconstruct_1_concurrent,prepare_data_list,life_time_image_reconstruct_4_concurrent,prepare_data_list_log,flt_est_cf_exp,life_time_est_cf_exp,life_time_image_reconstruct_1
#%%
def flt_img_exp(bin_array2,spectral_index,time_index,fignum):
    
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
    
def flt_img_ls(bin_array2,spectral_index,time_index,fignum):
    
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
    
    bin_Len,bin_list,time_list,bin_index_list,bin_log_list_partial,time_list_partial=prepare_data_list_log(frame_size,bin_array,bin_size,time_index,time_indices,time_line)
    
    start_time_0=timer()
    tau_4_array,r_4=life_time_image_reconstruct_4_concurrent(frame_size,bin_Len,bin_log_list_partial,time_list_partial,bin_index_list,n_cores)
    # tau_4_array=ndi.median_filter(tau_4_array,mode='nearest')
    tau_4_array1=sig.medfilt2d(tau_4_array)
    runtimeN1=(timer()-start_time_0)/60
    print(runtimeN1)
    
    ##%%
    plt.figure(fignum)
    # plt.subplot(121)
    # plt.imshow(tau_1_array,cmap='gray')
    # plt.colorbar()
    # plt.subplot(122)
    plt.imshow(tau_4_array1,cmap='gray')
    plt.colorbar()
    plt.show()
    plt.title('Curvefit-Exp fitting with $R^2$:%.3f'%r_4)
    
def resample_fn(bin_spec_x,decimate_factor_x):
    bin_spec_res_1_x=np.squeeze(sig.decimate(bin_spec_x,decimate_factor_x,ftype='fir'))
    return bin_spec_res_1_x

#%%
mypath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Normal/Row-1_Col-1_20230303'

# mypath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-1_20230214'
# List of responses of all tiles
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isdir(join(mypath, f))]
onlyfiles.sort()

# Mat file per tile
tile_file=3
matfile_list=listdir(onlyfiles[tile_file])
#iterable tile
matfile_list_path=join(onlyfiles[tile_file],matfile_list[0])#picking the mat file

start_time_0=timer()
mat_contents=h5py.File(matfile_list_path,'r+')
mat_contents_list=list(mat_contents.keys())
runtimeN0=(timer()-start_time_0)/60
print(runtimeN0)

start_time_0=timer()
bin_array_ref=mat_contents['bins_array_3']
frame_size_x_ref=mat_contents['frame_size_x']
hist_mode_ref=mat_contents['HIST_MODE']
binWidth_ref=mat_contents['binWidth']
runtimeN0=(timer()-start_time_0)/60
print(runtimeN0)

start_time_0=timer()
bin_array0=bin_array_ref[()]
frame_size=int(frame_size_x_ref[()])
hist_mode=int(hist_mode_ref[()])
binWidth=float(binWidth_ref[()])# time in ns
runtimeN0=(timer()-start_time_0)/60
print(runtimeN0)
#%% Data Parameters and Specifications

time_interval=binWidth
time_resolution=(binWidth*1000)/(2*2**hist_mode)# time unit in ps

bin_size=np.shape(bin_array0)
time_index=2

time_indices=np.arange(bin_size[time_index])
time_line=time_indices*time_interval# Time axis for fitting data

spectral_index=10 #stride over spectral dimension
spectral_span_sum=16
spectra_len=bin_size[-1]


#%% Moving sum over the spectral dimension
bin_array1=np.zeros_like(bin_array0)
start_time_0=timer()
for spec_i in range(bin_size[-1]):
    spectral_span=np.min([bin_size[3]-spec_i,spectral_span_sum])
    # print(spectral_span)
    bin_array1[:,:,:,spec_i]=np.sum(bin_array0[:,:,:,spec_i:spec_i+spectral_span],-1)
    
runtimeN0=(timer()-start_time_0)/60
print(runtimeN0)
#%% Band resampling
decimate_factor=5
bin_spec=bin_array1[255,255,14,:]
# bin_spec_res_1_bin=sig.decimate(bin_spec,decimate_factor,ftype='fir')
bin_spec_res_1_bin=resample_fn(bin_spec,decimate_factor)
spec_len_new=len(bin_spec_res_1_bin)
wave_spectrum=np.linspace(500, 780,bin_size[3])
# wave_samples_decimated=bin_size[3]//decimate_factor
wave_spectrum_new = np.linspace(500, 780, spec_len_new)

#%%
# spec_len=100
# start_spectrum=0
# stop_spectrum=bin_size[3]
# stop_spectrum_len=spec_len
bin_array2=np.zeros((bin_size[0],bin_size[1],bin_size[2],spec_len_new))
start_time_0=timer()
for loc_row1 in range(bin_size[0]):
    for loc_col1 in range(bin_size[1]):
        for time_bin in range(bin_size[2]):
            bin_spec1=bin_array1[loc_row1,loc_col1,time_bin,:]
            bin_spec_res_2_bin=resample_fn(bin_spec1,decimate_factor)
            bin_array2[loc_row1,loc_col1,time_bin,:]=bin_spec_res_2_bin
            
            
            # tau, r_squared=life_time_est_cf_exp(bin_resp,time_line,bin_size,time_indices,time_index)
            # tau_1_array[tau_row,tau_col]=tau
runtimeN0=(timer()-start_time_0)/60
print('Band resampling Loop %s'%runtimeN0)            
#%%

#Nested Loop
# start_time_0=timer()
# tau_1_array=np.zeros((frame_size,frame_size))
# for loc_row1 in range(bin_size[0]):
#     for loc_col1 in range(bin_size[1]):
#         bin_resp=bin_array1[loc_row1,loc_col1,:,spectral_index]
#         tau, r_squared=life_time_est_cf_exp(bin_resp,time_line,bin_size,time_indices,time_index)
#         tau_1_array[loc_row1,loc_col1]=tau
# runtimeN1=(timer()-start_time_0)/60
# print('Serialised Nested Loop %s'%runtimeN1)

# bin_array=bin_array2[:,:,:,spectral_index]
# bin_Len,bin_list,time_list,bin_index_list=prepare_data_list(frame_size,bin_array,bin_size,time_index,time_indices,time_line)

# #Vectorised Loop
# start_time_0=timer()
# tau_2_array=life_time_image_reconstruct_1(frame_size,bin_Len,bin_list,bin_index_list,time_line,bin_size,time_indices,time_index)
# runtimeN2=(timer()-start_time_0)/60
# print('Vectorised Loop %s'%runtimeN2)

#Multiprocessing Loop
start_time_0=timer()
flt_img_exp(bin_array2,spectral_index,time_index,10)
runtimeN3=(timer()-start_time_0)/60
print('Multiprocessing Loop %s'%runtimeN3)

#Multiprocessing Loop PD
start_time_0=timer()
flt_img_ls(bin_array2,spectral_index,time_index,20)
runtimeN4=(timer()-start_time_0)/60
print('Multiprocessing Loop PD %s'%runtimeN4)