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


bin_array=np.nan_to_num(np.real(np.log(bin_array)))

print('Curvefit startrd')
start_time_1=time.time()

tau_2=np.zeros((bin_size[spectral_index1],frame_size,frame_size))

for spectral_index in range(bin_size[spectral_index1]):      
    for loc_row1 in range(frame_size):
        for loc_col1 in range(frame_size):
            bin_resp1=bin_array[spectral_index,:,loc_row1,loc_col1]
            time_index_max1=bin_resp1.argmax()
            time_bin_selected1=bin_size[time_index]-time_index_max1-1
            time_bin_indices_selected1=time_indices[:-time_bin_selected1]
            bin_resp_selected1=bin_resp1[:-time_bin_selected1]# Look out for the 2nd dimension
            bin_resp_selected1=np.squeeze(bin_resp_selected1)
            # popt, pcov = curve_fit(lambda t, m, c: m * np.log(t) + c, time_bin_indices_selected1, bin_resp_selected1)
            popt, pcov = curve_fit(lambda t, m, c: m * t + c, time_bin_indices_selected1, bin_resp_selected1)
            m3 = popt[0]
            c3 = popt[1]
            tau_2[spectral_index,loc_row1,loc_col1]=m3

runtimeN1=(time.time()-start_time_1)/60
print('Curvefit finished')

print('Linear Regression started')

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()

tau_1=np.zeros((bin_size[spectral_index1],frame_size,frame_size))
for spectral_index in range(bin_size[spectral_index1]):   
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

runtimeN2=(time.time()-start_time_0)/60    

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()

print('Linear Regression finished')
print('polyfit started')
tau_3=np.zeros((bin_size[spectral_index1],frame_size,frame_size))
for spectral_index in range(bin_size[spectral_index1]):   
    for loc_row2 in range(frame_size):
        for loc_col2 in range(frame_size):
            bin_resp2=bin_array[spectral_index,:,loc_row2,loc_col2]
            time_index_max2=bin_resp2.argmax()
            time_bin_selected2=bin_size[time_index]-time_index_max2-1
            time_bin_indices_selected2=time_indices[:-time_bin_selected2]
            bin_resp_selected2=bin_resp2[:-time_bin_selected2]# Look out for the 2nd dimension
            bin_resp_selected2=np.squeeze(bin_resp_selected2)
            p2 = np.polyfit(time_bin_indices_selected2, bin_resp_selected2, 1)
            m2 = p2[0]
            c2 = p2[1]
            tau_3[spectral_index,loc_row1,loc_col1]=m2
    
runtimeN3=(time.time()-start_time_0)/60
print('polyfit finished')





# print(f'Values of m: {m1:5.3f}, {m2:5.3f}, {m3:5.3f}. Values of c: {c1:5.3f}, {c2:5.3f}, {c3:5.3f}')
# print(f'Values of m: {m3:5.3f}. Values of c:{c3:5.3f}')

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
spectral_index2=6
plt.figure(1)
plt.imshow(tau_1[spectral_index2,:,:],cmap='gray')
plt.colorbar()
plt.show()
plt.title('Linear Regression')

plt.figure(2)
plt.imshow(tau_2[spectral_index2,:,:],cmap='gray')
plt.colorbar()
plt.show()
plt.title('Curvefit')

plt.figure(3)
plt.imshow(tau_3[spectral_index2,:,:],cmap='gray')
plt.colorbar()
plt.show()
plt.title('Polyfit')