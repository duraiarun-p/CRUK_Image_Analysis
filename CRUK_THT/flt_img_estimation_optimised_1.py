#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 17:06:57 2023

@author: Arun PDRA, THT
"""

#%% Import Libraries

# import cupy as np

import numpy as np
import h5py

# import pdb
# import time
# import datetime

# import sys
# import os
from os import listdir
from os.path import join, isdir

import multiprocessing
# from  multiprocessing  import  Pool, Process
# import concurrent.futures

from timeit import default_timer as timer 

# from scipy import ndimage as ndi
from scipy import signal as sig

# from numba import cuda,jit

from matplotlib import pyplot as plt

from lifetime_estimate_lib_THT import life_time_image_reconstruct_1_concurrent,life_time_image_reconstruct_4_concurrent
#%%
def read_hd5(matfile_list_path):  
    mat_contents=h5py.File(matfile_list_path,'r+')
    # mat_contents = h5py.File(matfile_list_path, 'r+', driver='mpio', comm=MPI.COMM_WORLD)
    
    
    mat_contents_list=list(mat_contents.keys())
    
    bin_array_ref=mat_contents['bins_array_3']
    frame_size_x_ref=mat_contents['frame_size_x']
    hist_mode_ref=mat_contents['HIST_MODE']
    binWidth_ref=mat_contents['binWidth']
    
    bin_array0=bin_array_ref[()]
    frame_size=int(frame_size_x_ref[()])
    hist_mode=int(hist_mode_ref[()])
    binWidth=float(binWidth_ref[()])# time in ns
    
    return bin_array0,frame_size,hist_mode,binWidth,mat_contents_list

def vectorise_bin_array_4D_to_1D(frame_size,bin_array):
    
    bin_list=[]
    bin_log_list=[]
    bin_log_list_partial=[]
    bin_index_list=[]
    time_list=[]
    time_list_partial=[]
    
    count=0
    
    for loc_row1 in range(frame_size):
        for loc_col1 in range(frame_size):
            
            bin_resp=bin_array[loc_row1,loc_col1,:]
            # time_index_max=bin_resp.argmax()
            time_index_max=np.max(np.where(bin_resp==max(bin_resp)))
            # time_index_max=15
            count=count+1
            # if count == 643:
            #     print(count)
            #     pdb.set_trace()
                # time_index_max[time_index_max<8]=14 # Caused by low photon count
            # if time_index_max<14:
            #     time_index_max=14
            time_bin_selected=bin_size[time_index]-time_index_max-1
            # if time_bin_selected==0:
            #     time_bin_selected=1
            time_bin_indices_selected=time_indices[:-time_bin_selected]
            time_line_selected=time_line[time_bin_indices_selected]# x data for fitting
            bin_resp_selected=bin_resp[:-time_bin_selected]# Look out for the 2nd dimension
            bin_resp_selected=np.squeeze(bin_resp_selected)# y data for fitting
            bin_resp_selected=np.flip(bin_resp_selected)# Flipped for the real decay phenomenon
    
            bin_resp_selected_log=np.nan_to_num(np.log(bin_resp_selected),posinf=0, neginf=0) # log(y) data for fitting
            
            bin_index_list.append([loc_row1,loc_col1])
            bin_list.append(bin_resp_selected)
            bin_log_list.append(np.nan_to_num(np.log(bin_resp_selected),posinf=0, neginf=0))
            time_list.append(time_line_selected)
            time_list_partial.append(time_line_selected[:4])
            bin_log_list_partial.append(bin_resp_selected_log[:4])
            
    bin_Len=len(bin_list) # total number of pixel elements
    return bin_Len,bin_list,time_list,bin_index_list,bin_log_list,time_list_partial,bin_log_list_partial

#%%
n_cores = multiprocessing.cpu_count()


tile_file=3
time_index=2
spectral_index=100
spectral_span_sum=32

mypath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Normal/Row-1_Col-1_20230303'



onlyfiles = [join(mypath, f) for f in listdir(mypath) if isdir(join(mypath, f))]
onlyfiles.sort()
# Mat file per tile
# tile_file=3
matfile_list=listdir(onlyfiles[tile_file])
#iterable tile
matfile_list_path=join(onlyfiles[tile_file],matfile_list[0])#picking the mat file


start_time_0=timer()
bin_array0,frame_size,hist_mode,binWidth,mat_contents_list=read_hd5(matfile_list_path)
runtime_load=(timer()-start_time_0)/60


time_interval=binWidth

time_resolution=(binWidth*1000)/(2*2**hist_mode)# time unit in ps

bin_size=np.shape(bin_array0)
time_index=2

time_indices=np.arange(bin_size[time_index])
time_line=time_indices*time_interval# Time axis for fitting data


spectral_span_sum=16
# spectral_span_sum=16
bin_size=np.shape(bin_array0)

# bin_mean=np.mean(bin_array0)



# Best place for looping
spectral_index=1
#Similar to movsum
# bin_array=np.sum(bin_array0[:,:,:,spectral_index],-1)
bin_array=np.sum(bin_array0[:,:,:,spectral_index:spectral_index+spectral_span_sum],-1)
bin_int=bin_array[:,:,:]
bin_int_array=np.sum(bin_int,axis=time_index)
# bin_int_array=bin_int_array[:,:,-1]

#%%
# @cuda.jit(device=True)
# @jit(target_backend='cuda')
# @jit(nopython=True)


#%%
start_time_0=timer()
bin_Len,bin_list,time_list,bin_index_list,bin_log_list,time_list_partial,bin_log_list_partial=vectorise_bin_array_4D_to_1D(frame_size,bin_array)
runtime_vectorise1=(timer()-start_time_0)/60

#%%
# start_time_0=timer()
# pool = multiprocessing.Pool(processes=n_cores//2)
# # bin_Len,bin_list,time_list,bin_index_list,bin_log_list,time_list_partial,bin_log_list_partial=vectorise_bin_array_4D_to_1D(frame_size,bin_array)

# result_list = pool.map(vectorise_bin_array_4D_to_1D, frame_size,bin_array)
# runtime_vectorise2=(timer()-start_time_0)/60
# #%%
# p=Process(target=vectorise_bin_array_4D_to_1D, args=(frame_size,bin_array))
# p.start()
# p.join()

start_time_0=timer()
# tau_1_array=life_time_image_reconstruct_1(frame_size,bin_Len,bin_list,time_line,bin_index_list,n_cores)  
tau_1_array,r_1=life_time_image_reconstruct_1_concurrent(frame_size,bin_Len,bin_list,time_list,bin_index_list,n_cores)
# tau_1_array=sig.medfilt2d(tau_1_array)
# if (np.max(tau_1_array)-np.min(tau_1_array))/np.mean(tau_1_array)>2:
#     tau_1_array[tau_1_array>np.mean(tau_1_array)+((np.max(tau_1_array)-np.min(tau_1_array))/20)]=0
tau_1_array[tau_1_array>np.median(tau_1_array)*5]=0 # For visualisation
# tau_1_array=sig.medfilt2d(tau_1_array)
tau_1_array1=sig.medfilt2d(tau_1_array)
# tau_1_array1 = ma.masked_array(tau_1_array, bin_int_array_mask)
# tau_1_array1 = np.multiply(tau_1_array, bin_int_array_mask)
# tau_1,r_1=life_time_image_reconstruct_1_gpu(frame_size,bin_Len,bin_list,time_line)
runtimeN1=(timer()-start_time_0)/60

#%%

start_time_0=timer()
tau_4_array,r_4=life_time_image_reconstruct_4_concurrent(frame_size,bin_Len,bin_log_list_partial,time_list_partial,bin_index_list,n_cores)
# tau_4_array=ndi.median_filter(tau_4_array,mode='nearest')
tau_4_array1=sig.medfilt2d(tau_4_array)
# # tau_4_array1 = ma.masked_array(tau_4_array, bin_int_array_mask)
# tau_4_array1 = np.multiply(tau_4_array1, bin_int_array_mask)
# tau_4_array1=sig.medfilt2d(tau_4_array1)
runtimeN4=(timer()-start_time_0)/60


#%%
plt.figure(221)
# plt.subplot(121)
# plt.imshow(tau_1_array,cmap='gray')
# plt.colorbar()
# plt.subplot(122)
plt.imshow(tau_1_array1,cmap='gray')
plt.colorbar()
plt.show()
plt.title('Curvefit-Exp fitting with $R^2$:%.3f'%r_1)

plt.figure(224)
plt.imshow(tau_4_array1,cmap='gray')
plt.colorbar()
plt.show()
plt.title('Curvefit-LS fitting with $R_\mu^2$:%.3f'%r_4)

plt.figure(225)
plt.imshow(bin_int_array,cmap='gray')
plt.colorbar()
plt.show()
plt.title('Intensity')


