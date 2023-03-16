#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 13:43:46 2023

@author: arun
"""
#%% Import Libraries
import time
import datetime

import sys
import os

from matplotlib import pyplot as plt
from scipy.io import savemat
import h5py

import numpy as np
import random
from scipy.optimize import curve_fit
from scipy import stats

from multiprocessing import Process
from multiprocessing import Pool
import multiprocessing

import concurrent.futures
from traceback import print_exc

#%% Data Loading
matfile='/home/arun/Documents/PyWSPrecision/CRUK_THT/CRUK/HistMode_full_8bands_pixel_binning_inFW/PutSampleNameHere_Row_1_col_1/workspace.frame_1.mat'

mat_contents=h5py.File(matfile,'r')
mat_contents_list=list(mat_contents.keys())

bin_array_ref=mat_contents['bins_array_3']
frame_size_x_ref=mat_contents['frame_size_x']

bin_array=bin_array_ref[()]
frame_size=int(frame_size_x_ref[()])

#%% 

bin_size=np.shape(bin_array)
time_index=1

time_indices=np.arange(bin_size[time_index])

spectral_index1=0


# bin_array=np.nan_to_num(np.real(np.log(bin_array)),posinf=0, neginf=0)

print('Linear Regression started')

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()

def lifetime_fit(spectral_index,tau_1,frame_size,bin_array,bin_size,time_index):
    for loc_row in range(frame_size):
        for loc_col in range(frame_size):
            bin_resp=bin_array[spectral_index,:,loc_row,loc_col]
            time_index_max=bin_resp.argmax()
            time_bin_selected=bin_size[time_index]-time_index_max-1
            time_bin_indices_selected=time_indices[:-time_bin_selected]
            bin_resp_selected=bin_resp[:-time_bin_selected]# Look out for the 2nd dimension
            bin_resp_selected=np.squeeze(bin_resp_selected)
            p = stats.linregress(time_bin_indices_selected, bin_resp_selected)
            m1 = p[0]
            c1 = p[1]
            tau_1[spectral_index,loc_row,loc_col]=m1
    return tau_1

def lifetime_fit_1(bin_array_per_spectrum,frame_size,bin_size,time_index):
    tau_1_1=np.zeros((frame_size,frame_size))
    for loc_row in range(frame_size):
        for loc_col in range(frame_size):
            bin_resp=bin_array_per_spectrum[:,loc_row,loc_col]
            time_index_max=bin_resp.argmax()
            time_bin_selected=bin_size[time_index]-time_index_max-1
            
            time_bin_indices_selected=time_indices[:-time_bin_selected]     
            bin_resp_selected=bin_resp[:-time_bin_selected]# Look out for the 2nd dimension
            # time_bin_indices_selected=time_indices[-time_bin_selected-3:-time_bin_selected]     
            # bin_resp_selected=bin_resp[-time_bin_selected-3:-time_bin_selected]# Look out for the 2nd dimension
            
            bin_resp_selected=np.squeeze(bin_resp_selected)
            # p = stats.linregress(time_bin_indices_selected, bin_resp_selected)
            # m1 = p[0]
            # c1 = p[1]
            # tau_1_1[loc_row,loc_col]=m1
            # popt1, pcov1 = curve_fit(lambda t, a, b, c: a * np.exp(b * t) + c, time_bin_indices_selected, bin_resp_selected)
            popt1, pcov1 = curve_fit(lambda t, a, b, c: a * np.exp(b * t) + c, time_bin_indices_selected, bin_resp_selected,maxfev=50000)
            m3 = popt1[0]
            c3 = popt1[1]
            d3 = popt1[2]
            tau_1_1[loc_row,loc_col]=c3
            
    tau_1_1[tau_1_1>(np.mean(tau_1_1)*10)]=0
    return tau_1_1

tau_1=np.zeros((bin_size[spectral_index1],frame_size,frame_size))
# for spectral_index in range(bin_size[spectral_index1]):  
# for spectral_index in range(1):  
#     tau_1=lifetime_fit(tau_1,frame_size,bin_array,spectral_index,bin_size,time_index)

# procs = []
# for spectral_index in range(1):
#     proc = Process(target=lifetime_fit, args=(spectral_index,tau_1,frame_size,bin_array,bin_size,time_index))
#     procs.append(proc)
#     proc.start()

# # complete the processes
# for proc in procs:
#     proc.join()

n_cores = multiprocessing.cpu_count()
# pool = multiprocessing.Pool(n_cores)

# with pool:
#         # call the same function with different data in parallel
#         for result in pool.imap(lifetime_fit, range(1)),tau_1,frame_size,bin_array,bin_size,time_index:
#             # report the value to show progress
#             print(result)
# spectral_index=0
# bin_array_per_spectrum=bin_array[spectral_index,:,:,:]

# print(f'Values of m: {m1:5.3f}, {m2:5.3f}, {m3:5.3f}. Values of c: {c1:5.3f}, {c2:5.3f}, {c3:5.3f}')
# print(f'Values of m: {m3:5.3f}. Values of c:{c3:5.3f}')


# for spectral_index in range(1):
#     bin_array_per_spectrum=bin_array[spectral_index,:,:,:]
#     tau_1_1=lifetime_fit(bin_array_per_spectrum,spectral_index,frame_size,bin_size,time_index)

reslist=[]
resparlist=[]
spectra=bin_size[spectral_index1]
# spectra=1
    
with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
    for spectral_index in range(spectra):
        print('Parallel loop started')
        bin_array_per_spectrum=bin_array[spectral_index,:,:,:]
        res=executor.submit(lifetime_fit_1,bin_array_per_spectrum,frame_size,bin_size,time_index)
        reslist.append(res)
    print('Loop finished')      
    for resi in reslist:
        try:
            result = resi.result()
            resparlist.append(result)
        except:
            resparlist.append(None)
        print_exc()

# with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
#     for spectral_index in range(spectra):
#         print('Parallel loop started')
#         bin_array_per_spectrum=bin_array[spectral_index,:,:,:]
#         res=executor.submit(lifetime_fit_1,bin_array_per_spectrum,frame_size,bin_size,time_index)
#         reslist.append(res)
#     # print('Loop finished')      
#     # for resi in reslist:
#     #     try:
#         result = res.result()
#         resparlist.append(result)
#         # except:
#         # resparlist.append(None)
#         print_exc()

# for spectral_index1 in range(spectra):
#     bin_array_per_spectrum=bin_array[spectral_index1,:,:,:]
#     tau_1_1=lifetime_fit_1(bin_array_per_spectrum,frame_size,bin_size,time_index)
#     tau_1[spectral_index1,:,:]=tau_1_1


for spectral_index1 in range(spectra):
    tau_1[spectral_index1,:,:]=resparlist[spectral_index1]

runtimeN2=(time.time()-start_time_0)/60 

#%%
# x=time_bin_indices_selected
# y=bin_resp_selected
# ax = plt.axes()
# ax.scatter(x, y, c='gray', marker='o', edgecolors='k', s=18, label='Raw data')
# xlim = np.array(ax.get_xlim())
# xlim[0] = 0
# # ax.plot(xlim, 2 * xlim + 11, 'k--', label='True underlying relationship')
# ax.plot(xlim, m2 * xlim + c2, 'b', label='polyfit tool')
# ax.plot(xlim, m3 * xlim + c3, 'k', label='Curvefit tool')
# ax.set_title('Fluorecence decay')
# ax.set_xlabel(r'time index')
# ax.set_ylabel(r'intensity in counts')
# ax.set_xlim(xlim)
# ax.set_ylim(0)
# ax.legend(fontsize=8)
# plt.show()

#%%
spectral_index2=7
plt.figure(71)
plt.imshow(tau_1[spectral_index2,:,:],cmap='gray')
plt.colorbar()
plt.show()
plt.title('FLT')

# plt.figure(2)
# plt.imshow(result,cmap='gray')
# plt.colorbar()
# plt.show()
# plt.title('FLT-result')

# plt.figure(2)
# plt.imshow(tau_2[spectral_index2,:,:],cmap='gray')
# plt.colorbar()
# plt.show()
# plt.title('Curvefit')

# plt.figure(3)
# plt.imshow(tau_3[spectral_index2,:,:],cmap='gray')
# plt.colorbar()
# plt.show()
# plt.title('Polyfit')