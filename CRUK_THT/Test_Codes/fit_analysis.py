#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:01:47 2023

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
frame_size=frame_size_x_ref[()]

#%% 

def f(t,p):
    m=p[0]
    c=p[1]
    return  m * t + c

def f1(t,popt):
    a = popt[0]
    b = popt[1]
    c = popt[2]
    return a * np.exp(b * t) + c

bin_size=np.shape(bin_array)
time_index=1

time_indices=np.arange(bin_size[time_index])

spectral_index=1


# bin_array=np.log(bin_array)

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()

loc_row=np.random.randint(frame_size-1, size=(1))
loc_col=np.random.randint(frame_size-1, size=(1))

bin_resp=bin_array[spectral_index,:,loc_row,loc_col]
time_index_max=bin_resp.argmax()
time_bin_selected=bin_size[time_index]-time_index_max-1
time_bin_indices_selected=time_indices[:-time_bin_selected]
bin_resp_selected=bin_resp[:,:-time_bin_selected]# Look out for the 2nd dimension
bin_resp_selected=np.squeeze(bin_resp_selected)


# popt, pcov = curve_fit(lambda t, m, c: m * t + c, time_bin_indices_selected, bin_resp_selected)
popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(b * t) + c, time_bin_indices_selected, bin_resp_selected)
a = popt[0]
b = popt[1]
c = popt[2]
residuals = bin_resp_selected- f1(time_bin_indices_selected, popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((bin_resp_selected-np.mean(bin_resp_selected))**2)
r_squared = 1 - (ss_res / ss_tot)
#%%
plt.figure(101)
x=time_bin_indices_selected
y=bin_resp_selected
ax = plt.axes()
ax.scatter(x, y, c='gray', marker='o', edgecolors='k', s=18, label='Raw data')
xlim = np.array(ax.get_xlim())
xlim[0] = 0
# ax.plot(xlim, 2 * xlim + 11, 'k--', label='True underlying relationship')
# ax.plot(x, m2 * xlim + c2, 'b', label='polyfit tool')
ax.plot(x, a * np.exp(b * x) + c, 'k', label='Curvefit tool')
ax.set_title('Fluorecence decay (w/o) log \ R score: %.4f'%r_squared)
ax.set_xlabel(r'time index')
ax.set_ylabel(r'intensity in counts')
ax.set_xlim(xlim)
ax.set_ylim(0)
ax.legend(fontsize=8)
plt.show()




#%%

bin_resp_selected=np.log(bin_resp_selected)

p1 = stats.linregress(time_bin_indices_selected, bin_resp_selected)
m1 = p1[0]
c1 = p1[1]
residuals = bin_resp_selected- f(time_bin_indices_selected, p1)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((bin_resp_selected-np.mean(bin_resp_selected))**2)
r_squared1 = 1 - (ss_res / ss_tot)

# bin_resp_selected=np.log(bin_resp_selected)
p2 = np.polyfit(time_bin_indices_selected, bin_resp_selected, 1)
m2 = p2[0]
c2 = p2[1]
residuals = bin_resp_selected- f(time_bin_indices_selected, p2)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((bin_resp_selected-np.mean(bin_resp_selected))**2)
r_squared2 = 1 - (ss_res / ss_tot)

popt1, pcov1 = curve_fit(lambda t, m, c: m * t + c, time_bin_indices_selected, bin_resp_selected)
m3 = popt1[0]
c3 = popt1[1]
residuals = bin_resp_selected- f(time_bin_indices_selected, popt1)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((bin_resp_selected-np.mean(bin_resp_selected))**2)
r_squared3 = 1 - (ss_res / ss_tot)


# runtimeN0=(time.time()-start_time_0)/60

# # print(f'Values of m: {m1:5.3f}, {m2:5.3f}, {m3:5.3f}. Values of c: {c1:5.3f}, {c2:5.3f}, {c3:5.3f}')
# print(f'Values of m: {m3:5.3f}. Values of c:{c3:5.3f}')

#%%
plt.figure(100)
x=time_bin_indices_selected
y=bin_resp_selected
ax = plt.axes()
ax.scatter(x, y, c='gray', marker='o', edgecolors='k', s=18, label='Raw data')
xlim = np.array(ax.get_xlim())
xlim[0] = 0
# ax.plot(xlim, 2 * xlim + 11, 'k--', label='True underlying relationship')
ax.plot(x, m1 * x + c1, 'bo', label=['linear regression R^2 = ' + str(r_squared1)])
ax.plot(x, m2 * x + c2, 'gx', label='polyfit ')
ax.plot(x, m3 * x + c3, 'r-', label='Curvefit ')
ax.set_title('Fluorecence decay (w) log')
ax.set_xlabel(r'time index')
ax.set_ylabel(r'intensity in counts')
ax.set_xlim(xlim)
ax.set_ylim(0)
ax.legend(fontsize=8)
plt.show()

