#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:12:07 2023

@author: arun
"""
#%% Import Libraries
import pdb
import time
import datetime

import sys
import os

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

import multiprocessing

import concurrent.futures
from traceback import print_exc

from timeit import default_timer as timer 

# from jax.config import config
# config.update("jax_enable_x64", True)

# from jaxfit import CurveFit
# import jax.numpy as jnp
from lifetime_estimate_lib import life_time_image_reconstruct_1_concurrent_1

n_cores = multiprocessing.cpu_count()

#%%
matfile='/home/arun/Documents/PyWSPrecision/CRUK_Image_Analysis/CRUK_THT/CRUK/Row_1_Col_1_N/workspace.frame_1.mat'

mat_contents=h5py.File(matfile,'r')
mat_contents_list=list(mat_contents.keys())

bin_array_ref=mat_contents['bins_array_3']
frame_size_x_ref=mat_contents['frame_size_x']
hist_mode_ref=mat_contents['HIST_MODE']
binWidth_ref=mat_contents['binWidth']

bin_array0=bin_array_ref[()]
frame_size=int(frame_size_x_ref[()])
hist_mode=int(hist_mode_ref[()])
binWidth=float(binWidth_ref[()])# time in ns

#%% Array Slicing based on spectrum/wavelength and parameter selection
time_interval=binWidth

time_resolution=(binWidth*1000)/(2*2**hist_mode)# time unit in ps

bin_size=np.shape(bin_array0)
time_index=2

time_indices=np.arange(bin_size[time_index])
time_line=time_indices*time_interval# Time axis for fitting data

spectral_index=100 'stride over spectral dimension'


spectral_span_sum=32
bin_size=np.shape(bin_array0)

# bin_mean=np.mean(bin_array0)

#Similar to movsum
bin_array=np.sum(bin_array0[:,:,:,spectral_index:spectral_index+spectral_span_sum],-1)

bin_int=bin_array[:,:,:]
# bin_int_array_4_mask=bin_array[:,:,0]

bin_int_array=np.cumsum(bin_int,axis=time_index)
bin_int_array=bin_int_array[:,:,-1]

bin_int_array1=filters.gaussian(bin_int_array, sigma=30)

# Extract this threshold in the first file then carry over to the subsequent tiles
bin_mask_thresh=np.mean(bin_int_array)*.75
bin_int_array_mask=np.zeros_like(bin_int_array)
bin_int_array_mask[bin_int_array1>bin_mask_thresh]=1
bin_int_array_mask = ndi.binary_fill_holes(bin_int_array_mask)
bin_int_array_mask = util.invert(bin_int_array_mask)
#%%

bin_list=[]
bin_log_list=[]
bin_log_list_partial=[]
bin_index_list=[]
time_list=[]
time_list_partial=[]

count=0

start_time_0=timer()

for loc_row1 in range(frame_size):
    for loc_col1 in range(frame_size):
        # bin_resp=bin_array[spectral_index,:,loc_row1,loc_col1]
        # bin_resp=np.squeeze(bin_array[loc_row1,loc_col1,:,spectral_index])
        # bin_resp=bin_array[loc_row1,loc_col1,:,spectral_index]
        bin_resp=bin_array[loc_row1,loc_col1,:]
        # time_index_max=bin_resp.argmax()
        time_index_max=np.max(np.where(bin_resp==max(bin_resp)))
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
        # bin_log_list.append(np.nan_to_num(np.log(bin_resp_selected),posinf=0, neginf=0))
        time_list.append(time_line_selected)
        time_list_partial.append(time_line_selected[:4])
        # bin_log_list_partial.append(bin_resp_selected_log[:4])
        
runtimeN0=(timer()-start_time_0)/60
        
bin_Len=len(bin_list) # total number of pixel elements

#%%
# cf = CurveFit()

# def decay_fn(t,popt):
#     a = popt[0]
#     b = popt[1]
#     c = popt[2]
#     return a * jnp.exp(b * -t) - c

# def jax_exp(t, a, b, c): # fit function
# # 	return jnp.exp(a * x) + b
#     return a * jnp.exp(b * -t) - c


# # popt, pcov = cf.curve_fit(decay_fn, x, y)

# def flt_est_jax_exp(bin_resp_selected1,time_line_selected1):
#     # popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(b * -t) - c, bin_resp_selected, time_line_selected,maxfev=50000)
#     # popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(b * -t) - c, time_line_selected, bin_resp_selected,maxfev=50000)
#     poptx, pcovx = cf.curve_fit(jax_exp, time_line_selected1, bin_resp_selected1)
#     a = poptx[0]
#     b = poptx[1]
#     c = poptx[2]
#     residuals = bin_resp_selected1- decay_fn(time_line_selected1, poptx)
#     ss_res = np.sum(residuals**2)
#     ss_tot = np.sum((bin_resp_selected1-np.mean(bin_resp_selected1))**2)
#     r_squared = 1 - (ss_res / ss_tot)
#     return abs(b), r_squared


# def life_time_image_reconstruct_1(frame_size,bin_Len,bin_list,time_list,bin_index_list):
#     tau_1_array=np.zeros((frame_size,frame_size))    
#     tau_1=np.zeros((bin_Len,1))
#     r_1=np.zeros((bin_Len,1))
#     # for bin_index in range(bin_Len):
#     for bin_index in range(1):
#         print(bin_index)
#         bin_resp=bin_list[bin_index]
#         time_line=time_list[bin_index]
#         tau, r_squared=flt_est_jax_exp(bin_resp,time_line)
#         tau_1[bin_index]=tau
#         r_1[bin_index]=r_squared
#         tau_row=bin_index_list[bin_index][0]
#         tau_col=bin_index_list[bin_index][1]
#         tau_1_array[tau_row,tau_col]=tau
#     # tau_1_array[tau_1_array>(np.mean(tau_1_array)*10)]=0
#     return tau_1_array,np.mean(r_1)


# def life_time_image_reconstruct_1_concurrent(frame_size,bin_Len,bin_list,time_list,bin_index_list,n_cores):
#     tau_1_array=np.zeros((frame_size,frame_size))    
#     tau_1=np.zeros((bin_Len,1))
#     r_1=np.zeros((bin_Len,1))
#     res=[]
#     resparlist=[]
#     reslist=[]
#     with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:       
#         futures = {executor.submit(flt_est_jax_exp,bin_resp,time_line) for bin_resp,time_line in zip(bin_list,time_list)}
#         for fut in concurrent.futures.as_completed(futures):
#             result = fut.result()
#             resparlist.append(result)
#         for bin_index in range(bin_Len):
#             tau_row=bin_index_list[bin_index][0]
#             tau_col=bin_index_list[bin_index][1]
#             tau_1[bin_index]=resparlist[bin_index][0]
#             r_1[bin_index]=resparlist[bin_index][1]
#             tau_1_array[tau_row,tau_col]=resparlist[bin_index][0]
#     return tau_1_array,np.mean(r_1)

#%%

bin_Len=len(bin_list) # total number of pixel elements

start_time_0=timer()
# tau_1_array,r_1=life_time_image_reconstruct_1(frame_size,bin_Len,bin_list,time_line,bin_index_list)
tau_1_array,r_1=life_time_image_reconstruct_1_concurrent(frame_size,bin_Len,bin_list,time_list,bin_index_list,n_cores)
tau_1_array[tau_1_array>np.median(tau_1_array)*20]=0 # For visualisation
tau_1_array=sig.medfilt2d(tau_1_array)
tau_1_array1 = ma.masked_array(tau_1_array, bin_int_array_mask)
runtimeN1=(timer()-start_time_0)/60

#%%
plt.figure(21)
# plt.subplot(121)
# plt.imshow(tau_1_array,cmap='gray')
# plt.colorbar()
# plt.subplot(122)
plt.imshow(tau_1_array1,cmap='gray')
plt.colorbar()
plt.show()
plt.title('Curvefit-Exp fitting with $R^2$:%.3f'%r_1)