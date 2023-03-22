#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:14:31 2023

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
# matfile='/home/arun/Documents/PyWSPrecision/CRUK_Image_Analysis/CRUK_THT/CRUK/HistMode_full_8bands_pixel_binning_inFW/PutSampleNameHere_Row_1_col_1/workspace.frame_1.mat'

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
#%% Condiioning
bin_size=np.shape(bin_array0)

bin_mean=np.mean(bin_array0)

spec_indices=np.linspace(start = 1, stop = bin_size[-1], num = bin_size[-1])
# bin_array0=bin_array0-bin_mean

# spectral_span_sum=32
#%%
# spectral_index=100

# bin_array_32=np.sum(bin_array0[:,:,:,spectral_index:spectral_index+32],-1)
# bin_array_64=np.sum(bin_array0[:,:,:,spectral_index:spectral_index+64],-1)

bin_resp_spec_0=np.zeros((bin_size[-1],1))
bin_resp_spec_8=np.zeros((bin_size[-1],1))
bin_resp_spec_16=np.zeros((bin_size[-1],1))
bin_resp_spec_32=np.zeros((bin_size[-1],1))
bin_resp_spec_64=np.zeros((bin_size[-1],1))
for spec_i in range(bin_size[-1]):
    # bin_resp_spec[spec_i]=np.sum(bin_array0[:,:,:,spec_i])
    bin_resp_spec_0[spec_i]=np.sum(bin_array0[:,:,15,spec_i])
    spectral_span=np.min([bin_size[3]-spec_i,8]) 
    bin_resp_spec_8[spec_i]=np.sum(bin_array0[:,:,15,spec_i:spec_i+spectral_span])
    spectral_span=np.min([bin_size[3]-spec_i,16]) 
    bin_resp_spec_16[spec_i]=np.sum(bin_array0[:,:,15,spec_i:spec_i+spectral_span])
    spectral_span=np.min([bin_size[3]-spec_i,32]) 
    bin_resp_spec_32[spec_i]=np.sum(bin_array0[:,:,15,spec_i:spec_i+spectral_span])
    spectral_span=np.min([bin_size[3]-spec_i,64]) 
    bin_resp_spec_64[spec_i]=np.sum(bin_array0[:,:,15,spec_i:spec_i+spectral_span])
    
#%% 
plt.figure(400)
    # x=time_bin_indices_selected
    # y=bin_resp_selected
ax = plt.axes()
# ax.scatter(spec_indices,bin_resp_spec_32, c='gray', marker='o', edgecolors='k', s=18, label='Raw data')
ax.plot(spec_indices,bin_resp_spec_32,'c', label='sliding window = 32')
ax.plot(spec_indices,bin_resp_spec_64,'g', label='sliding window = 64')
ax.plot(spec_indices,bin_resp_spec_0,'k', label='sliding window = 0')
ax.plot(spec_indices,bin_resp_spec_8,'b', label='sliding window = 8')
ax.plot(spec_indices,bin_resp_spec_16,'r', label='sliding window = 16')
# xlim = np.array(ax.get_xlim())
# xlim[0] = 0
# ax.plot(xlim, 2 * xlim + 11, 'k--', label='True underlying relationship')
# ax.plot(x, m2 * xlim + c2, 'b', label='polyfit tool')
# ax.plot(x, f1(x,popt), 'k', label=labelstring)
ax.set_title(r'Fluorecence spectrum')
ax.set_xlabel(r'Spectrum Index')
ax.set_ylabel(r'Intensity (counts)')
ax.set_xlim(-0.1)
ax.set_ylim(-0.1)
ax.legend(fontsize=8)
plt.show()

plt.figure(401)
    # x=time_bin_indices_selected
    # y=bin_resp_selected
ax = plt.axes()
# ax.scatter(spec_indices,bin_resp_spec_32, c='gray', marker='o', edgecolors='k', s=18, label='Raw data')
ax.plot(spec_indices[45:],bin_resp_spec_32[45:],'c', label='sliding window = 32')
ax.plot(spec_indices[45:],bin_resp_spec_64[45:],'g', label='sliding window = 64')
ax.plot(spec_indices[45:],bin_resp_spec_0[45:],'k', label='sliding window = 0')
ax.plot(spec_indices[45:],bin_resp_spec_8[45:],'b', label='sliding window = 8')
ax.plot(spec_indices[45:],bin_resp_spec_16[45:],'r', label='sliding window = 16')
# xlim = np.array(ax.get_xlim())
# xlim[0] = 0
# ax.plot(xlim, 2 * xlim + 11, 'k--', label='True underlying relationship')
# ax.plot(x, m2 * xlim + c2, 'b', label='polyfit tool')
# ax.plot(x, f1(x,popt), 'k', label=labelstring)
ax.set_title(r'Fluorecence spectrum')
ax.set_xlabel(r'Spectrum Index')
ax.set_ylabel(r'Intensity (counts)')
ax.set_xlim(-0.1)
ax.set_ylim(-0.1)
ax.legend(fontsize=8)
plt.show()

plt.figure(402)
    # x=time_bin_indices_selected
    # y=bin_resp_selected
ax = plt.axes()
# ax.scatter(spec_indices,bin_resp_spec_32, c='gray', marker='o', edgecolors='k', s=18, label='Raw data')
# ax.plot(spec_indices[45:],bin_resp_spec_32[45:],'b', label='sliding window = 32')
# ax.plot(spec_indices[45:],bin_resp_spec_64[45:],'r', label='sliding window = 64')
ax.plot(spec_indices[45:],bin_resp_spec_0[45:],'k', label='sliding window = 0')
# xlim = np.array(ax.get_xlim())
# xlim[0] = 0
# ax.plot(xlim, 2 * xlim + 11, 'k--', label='True underlying relationship')
# ax.plot(x, m2 * xlim + c2, 'b', label='polyfit tool')
# ax.plot(x, f1(x,popt), 'k', label=labelstring)
ax.set_title(r'Fluorecence spectrum')
ax.set_xlabel(r'Spectrum Index')
ax.set_ylabel(r'Intensity (counts)')
ax.set_xlim(-0.1)
ax.set_ylim(-0.1)
ax.legend(fontsize=8)
plt.show()