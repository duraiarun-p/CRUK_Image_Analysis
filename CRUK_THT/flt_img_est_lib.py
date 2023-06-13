#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 20:47:41 2023

@author: Arun PDRA, THT
"""

import sys
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

import concurrent.futures
import multiprocessing

from lifetime_estimate_lib_THT import life_time_image_reconstruct_1_concurrent,prepare_data_list,prepare_data_list_wo_flip
#%%

def resample_fn(bin_spec_x,decimate_factor_x):
    bin_spec_res_1_x=np.squeeze(sig.decimate(np.squeeze(bin_spec_x),decimate_factor_x,ftype='fir'))
    return bin_spec_res_1_x

def resample_fn_2(bin_spec_x,decimate_factor_x):
    bin_spec_res_1_x=np.squeeze(sig.resample(bin_spec_x,len(bin_spec_x)//(decimate_factor_x),domain='time'))
    return bin_spec_res_1_x.clip(min=0)

def flt_img_exp(bin_array2,spectral_index,time_index,bin_size,frame_size,time_indices,time_line):
    
    bin_array=bin_array2[:,:,:,spectral_index]# only for spectral span = 1
    # bin_array=bin_int
    bin_int_array=np.sum(bin_array,axis=time_index)
    
    bin_int_array1=filters.gaussian(bin_int_array, sigma=10)

    bin_mask_thresh=np.mean(bin_int_array1)*.75
    bin_int_array_mask=np.zeros_like(bin_int_array1)
    bin_int_array_mask[bin_int_array1>bin_mask_thresh]=1
        # bin_int_array_mask = filters.sobel(bin_int_array_mask)
        # bin_int_array_mask=area_closing(bin_int_array_mask,area_threshold=64)
    bin_int_array_mask = ndi.binary_fill_holes(bin_int_array_mask)
 
    n_cores=multiprocessing.cpu_count()
    
    bin_Len,bin_list,time_list,bin_index_list=prepare_data_list(frame_size,bin_array,bin_size,time_index,time_indices,time_line)
    
    start_time_0=timer()
    # tau_1_array=life_time_image_reconstruct_1(frame_size,bin_Len,bin_list,time_line,bin_index_list,n_cores)  
    tau_1_array,r_1=life_time_image_reconstruct_1_concurrent(frame_size,bin_Len,bin_list,time_list,bin_index_list,n_cores)
    # tau_1_array=sig.medfilt2d(tau_1_array)
    # if (np.max(tau_1_array)-np.min(tau_1_array))/np.mean(tau_1_array)>2:
    #     tau_1_array[tau_1_array>np.mean(tau_1_array)+((np.max(tau_1_array)-np.min(tau_1_array))/20)]=0
    #tau_1_array[tau_1_array>np.median(tau_1_array)*50]=0 # For visualisation # Commented to analyse full core
    # tau_1_array=sig.medfilt2d(tau_1_array)
    tau_1_array1=sig.medfilt2d(tau_1_array)
    # tau_1_array1 = ma.masked_array(tau_1_array, bin_int_array_mask)
    # tau_1_array1 = np.multiply(tau_1_array, bin_int_array_mask)
    # tau_1,r_1=life_time_image_reconstruct_1_gpu(frame_size,bin_Len,bin_list,time_line)
    runtimeN1=(timer()-start_time_0)/60
    print('Lifetime reconstruction time %s'%runtimeN1)
    return tau_1_array1,bin_int_array

def flt_img_exp_wo_flip(bin_array2,spectral_index,time_index,bin_size,frame_size,time_indices,time_line):
    
    bin_array=bin_array2[:,:,:,spectral_index]# only for spectral span = 1
    # bin_array=bin_int
    bin_int_array=np.sum(bin_array,axis=time_index)
    
    bin_int_array1=filters.gaussian(bin_int_array, sigma=10)

    bin_mask_thresh=np.mean(bin_int_array1)*.75
    bin_int_array_mask=np.zeros_like(bin_int_array1)
    bin_int_array_mask[bin_int_array1>bin_mask_thresh]=1
        # bin_int_array_mask = filters.sobel(bin_int_array_mask)
        # bin_int_array_mask=area_closing(bin_int_array_mask,area_threshold=64)
    bin_int_array_mask = ndi.binary_fill_holes(bin_int_array_mask)
 
    n_cores=multiprocessing.cpu_count()
    
    bin_Len,bin_list,time_list,bin_index_list=prepare_data_list_wo_flip(frame_size,bin_array,bin_size,time_index,time_indices,time_line)
    
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
    print('Lifetime reconstruction time %s'%runtimeN1)
    return tau_1_array1,bin_int_array1



#%%
def flt_per_tile(matfile_list_path,decimate_factor,spec_resampled,spec_truncated):
    
    # onlyfiles = [join(mypath, f) for f in listdir(mypath) if isdir(join(mypath, f))]
    # onlyfiles.sort()
    
    # # Mat file per tile
    # # tile_file=3
    # matfile_list=listdir(onlyfiles[tile_file])
    # #iterable tile
    # matfile_list_path=join(onlyfiles[tile_file],matfile_list[0])#picking the mat file
    
    start_time_0=timer()
    mat_contents=h5py.File(matfile_list_path,'r+')
    mat_contents_list=list(mat_contents.keys())
    runtimeN0=(timer()-start_time_0)/60
    print('Load Mat file time %s'%runtimeN0)
    
    start_time_0=timer()
    bin_array_ref=mat_contents['bins_array_3']
    frame_size_x_ref=mat_contents['frame_size_x']
    hist_mode_ref=mat_contents['HIST_MODE']
    binWidth_ref=mat_contents['binWidth']
    runtimeN0=(timer()-start_time_0)/60
    print('Extract content from Hdf5 time %s'%runtimeN0)
    
    start_time_0=timer()
    bin_array0=bin_array_ref[()]
    frame_size=int(frame_size_x_ref[()])
    hist_mode=int(hist_mode_ref[()])
    binWidth=float(binWidth_ref[()])# time in ns
    runtimeN0=(timer()-start_time_0)/60
    print('Ref from Hdf5 time %s'%runtimeN0)
    #%% Data Parameters and Specifications
    
    time_interval=binWidth
    time_resolution=(binWidth*1000)/(2*2**hist_mode)# time unit in ps
    
    bin_size=np.shape(bin_array0)
    time_index=2
    
    time_indices=np.arange(bin_size[time_index])
    time_line=time_indices*time_interval# Time axis for fitting data
    
    # spectral_index=10 #stride over spectral dimension
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
    print('Moving sum time %s'%runtimeN0)
    #%% Band resampling
    # decimate_factor=5
    bin_spec=bin_array1[255,255,14,:spec_truncated]
    # bin_spec_res_1_bin=sig.decimate(bin_spec,decimate_factor,ftype='fir')
    bin_spec_res_1_bin=resample_fn_2(bin_spec,decimate_factor)
    spec_len_new=len(bin_spec_res_1_bin)
    wave_spectrum=np.linspace(500, 780,bin_size[3])
    # wave_samples_decimated=bin_size[3]//decimate_factor
    wave_spectrum_new = np.linspace(500, wave_spectrum[spec_truncated], spec_len_new)
    
    #%% Freeing up memory 
    del bin_array0
    
    #%% changing 4d to 1d array
    # spec_len=100
    # start_spectrum=0
    # stop_spectrum=bin_size[3]
    # stop_spectrum_len=spec_len
    # bin_array2_list=[]
    # bin_index2_list=[]
    # start_time_0=timer()
    
    # for loc_row1 in range(bin_size[0]):
    #     for loc_col1 in range(bin_size[1]):
    #         for time_bin in range(bin_size[2]):
    #             bin_spec1=bin_array1[loc_row1,loc_col1,time_bin,:spec_resampled]
                
    #             bin_index2_list.append([loc_row1,loc_col1,time_bin])        
    #             bin_array2_list.append(bin_spec1)
    #             # bin_spec_res_2_bin=resample_fn(bin_spec1,decimate_factor)
    #             # bin_array2[loc_row1,loc_col1,time_bin,:]=bin_spec_res_2_bin
    
    
    # runtimeN0=(timer()-start_time_0)/60
    # print('Band resampling list Loop %s'%runtimeN0)  
    #%% Band resampling - consider GPU
    # bin_array2_list_len=len(bin_array2_list)
    # n_cores=multiprocessing.cpu_count()
    # # bin_array2=np.zeros((bin_size[0],bin_size[1],bin_size[2],spec_len_new))
    # resparlist=[]#
    # start_time_0=timer()
    # with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:       
    #     futures = {executor.submit(resample_fn,bin_spec_x,decimate_factor) for bin_spec_x in bin_array2_list}
    #     for fut in concurrent.futures.as_completed(futures):
    #         result = fut.result()
    #         resparlist.append(result)
    # runtimeN0=(timer()-start_time_0)/60
    # print('Band resampling Loop %s'%runtimeN0)
    #%%
    # bin_array2=np.zeros((bin_size[0],bin_size[1],bin_size[2],spec_len_new))
    # start_time_0=timer()
    # for bin_index2 in range(bin_array2_list_len):
    #     time_bin=bin_index2_list[bin_index2][2]
    #     bin_array2[bin_index2_list[bin_index2][0],bin_index2_list[bin_index2][1],bin_index2_list[bin_index2][2],:]=resparlist[bin_index2]
    # runtimeN0=(timer()-start_time_0)/60
    # print('Band resampled array Loop %s'%runtimeN0)
    # bin_array2=bin_array1[:,:,:,:spec_resampled]
    #spec_index=np.round(np.linspace(0,spec_truncated-1,spec_resampled)).astype(int)
    #spec_len_new=len(spec_index)
    start_time_0=timer()
    bin_array2=np.zeros((bin_size[0],bin_size[1],bin_size[2],spec_len_new))
    for loc_row1 in range(bin_size[0]):
        for loc_col1 in range(bin_size[1]):
            for time_bin in range(bin_size[2]):
                bin_spec1=bin_array1[loc_row1,loc_col1,time_bin,:spec_truncated]
                
                # bin_spec_res_2_bin=resample_fn(bin_spec1,decimate_factor)# Decimation
                # choose bin based on selection
                # bin_spec_res_2_bin=bin_spec1[spec_index]
                bin_spec_res_2_bin=resample_fn_2(bin_spec1,decimate_factor)# Resampling
                bin_array2[loc_row1,loc_col1,time_bin,:]=bin_spec_res_2_bin
    runtimeN0=(timer()-start_time_0)/60
    print('Band resampling Loop %s'%runtimeN0)
    #%%
    spec_resampled_new=np.arange(0,spec_resampled,1)# change it to 2 for Every 2nd compnent
    # spec_resampled_new=np.arange(0,spec_resampled//2,1)# FIrst 10 compnent
    wave_spectrum_new_select=wave_spectrum_new[spec_resampled_new]
    intensity=np.zeros((bin_size[0],bin_size[1],len(spec_resampled_new)))
    flt=np.zeros((bin_size[0],bin_size[1],len(spec_resampled_new)))
    start_time_0=timer()
    # for spectral_index in range(spec_resampled):   
    for spectral_index in range(len(spec_resampled_new)): 
        print(spectral_index)
        spectral_index_new=spec_resampled_new[spectral_index]
        tau_1_array1,bin_int_array1=flt_img_exp(bin_array2,spectral_index_new,time_index,bin_size,frame_size,time_indices,time_line)
        intensity[:,:,spectral_index]=bin_int_array1
        flt[:,:,spectral_index]=tau_1_array1
    runtimeN3=(timer()-start_time_0)/60
    print('Image built time %s'%runtimeN3)
    
    return intensity,flt,wave_spectrum,wave_spectrum_new,wave_spectrum_new_select
