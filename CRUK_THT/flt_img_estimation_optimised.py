#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:02:21 2023

@author: Arun PDRA, THT
"""

#%% Import Libraries

# from mpi4py import MPI
# from numba import njit, prange
from numba import cuda,jit,njit
import h5py

import pdb
import time
import datetime

import sys
import os
from os import listdir
from os.path import isfile, join, isdir

import multiprocessing
# from  multiprocessing  import  Pool
import concurrent.futures

from timeit import default_timer as timer 

import numpy as np
#%% Defintions
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




#%% Data loading
# rank = MPI.COMM_WORLD.rank

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


# with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:       
#     futures = {executor.submit(read_hd5,matfile_list_path)}
#     for fut in concurrent.futures.as_completed(futures):
#         result = fut.result()

time_interval=binWidth

time_resolution=(binWidth*1000)/(2*2**hist_mode)# time unit in ps

bin_size=np.shape(bin_array0)
time_index=2

time_indices=np.arange(bin_size[time_index])
time_line=time_indices*time_interval# Time axis for fitting data

spectral_index=1
spectral_span_sum=16
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

#%% An attempt to parallelize the loops
# @njit(parallel=True)
# @cuda.jit(device=True)
# @jit(target_backend='cuda')
# @jit(nopython=True)
# @cuda.jit
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
start_time_0=timer()
bin_Len,bin_list,time_list,bin_index_list,bin_log_list,time_list_partial,bin_log_list_partial=vectorise_bin_array_4D_to_1D(frame_size,bin_array)
runtime_vectorise=(timer()-start_time_0)/60