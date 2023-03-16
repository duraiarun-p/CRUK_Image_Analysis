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
frame_size=int(frame_size_x_ref[()])

#%% 

bin_size=np.shape(bin_array)
time_index=1

time_indices=np.arange(bin_size[time_index])

spectral_index=-1


# bin_array=np.nan_to_num(np.real(np.log(bin_array)))

def f1(t,popt):
    a = popt[0]
    b = popt[1]
    c = popt[2]
    return a * np.exp(b * t) + c
    

print('Curvefit started')
start_time_1=time.time()

tau_2=np.zeros((frame_size,frame_size))

r2_2=[]

for loc_row1 in range(frame_size):
    for loc_col1 in range(frame_size):
        bin_resp1=bin_array[spectral_index,:,loc_row1,loc_col1]
        time_index_max1=bin_resp1.argmax()
        time_bin_selected1=bin_size[time_index]-time_index_max1-1
        time_bin_indices_selected1=time_indices[:-time_bin_selected1]
        bin_resp_selected1=bin_resp1[:-time_bin_selected1]# Look out for the 2nd dimension
        bin_resp_selected1=np.squeeze(bin_resp_selected1)
        # popt, pcov = curve_fit(lambda t, m, c: m * np.log(t) + c, time_bin_indices_selected1, bin_resp_selected1)
        # popt, pcov = curve_fit(lambda t, m, c: m * t + c, time_bin_indices_selected1, bin_resp_selected1)
        # popt, pcov = curve_fit(lambda t, m, c: m * np.exp(t) - c, time_bin_indices_selected1, bin_resp_selected1)
        # m3 = popt[0]
        # c3 = popt[1]
        # p0=[2,0.4,24]
        # print('%s \ %s'%(loc_row1,loc_col1))
        popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(b * t) + c, time_bin_indices_selected1, bin_resp_selected1,maxfev=50000)
        a = popt[0]
        b = popt[1]
        c = popt[2]
        tau_2[loc_row1,loc_col1]=b
        
        residuals = bin_resp_selected1- f1(time_bin_indices_selected1, popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((bin_resp_selected1-np.mean(bin_resp_selected1))**2)
        r_squared = 1 - (ss_res / ss_tot)
        r2_2.append(r_squared)

# itemindex = np.where(tau_2 == np.max(tau_2))
tau_2[tau_2>(np.mean(tau_2)*10)]=0

runtimeN1=(time.time()-start_time_1)/60
print('Curvefit finished')


def f(t,p):
    m=p[0]
    c=p[1]
    return  m * t + c

# bin_array=np.nan_to_num(np.real(np.log(bin_array)))

print('Linear Regression started')

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()
r2_1=[]
tau_1=np.zeros((frame_size,frame_size))
for loc_row in range(frame_size):
    for loc_col in range(frame_size):
        bin_resp=bin_array[spectral_index,:,loc_row,loc_col]
        time_index_max=bin_resp.argmax()
        time_bin_selected=bin_size[time_index]-time_index_max-1
        time_bin_indices_selected=time_indices[:-time_bin_selected]
        bin_resp_selected=bin_resp[:-time_bin_selected]# Look out for the 2nd dimension
        bin_resp_selected=np.squeeze(bin_resp_selected)
        bin_resp_selected=np.nan_to_num(np.log(bin_resp_selected),posinf=0, neginf=0)
        p = stats.linregress(time_bin_indices_selected, bin_resp_selected)
        m1 = p[0]
        c1 = p[1]
        tau_1[loc_row,loc_col]=m1
        
        residuals = bin_resp_selected- f(time_bin_indices_selected, p)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((bin_resp_selected-np.mean(bin_resp_selected))**2)
        r_squared = 1 - (ss_res / ss_tot)
        r2_1.append(r_squared)
    
runtimeN2=(time.time()-start_time_0)/60

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()

print('Linear Regression finished')
print('polyfit started')
tau_3=np.zeros((frame_size,frame_size))
r2_3=[]
for loc_row2 in range(frame_size):
    for loc_col2 in range(frame_size):
        bin_resp2=bin_array[spectral_index,:,loc_row2,loc_col2]
        time_index_max2=bin_resp2.argmax()
        time_bin_selected2=bin_size[time_index]-time_index_max2-1
        time_bin_indices_selected2=time_indices[:-time_bin_selected2]
        bin_resp_selected2=bin_resp2[:-time_bin_selected2]# Look out for the 2nd dimension
        bin_resp_selected2=np.squeeze(bin_resp_selected2)
        bin_resp_selected2=np.nan_to_num(np.log(bin_resp_selected2),posinf=0, neginf=0)
        p2 = np.polyfit(time_bin_indices_selected2, bin_resp_selected2, 1,w=np.sqrt(bin_resp_selected2))
        m2 = p2[0]
        c2 = p2[1]
        tau_3[loc_row2,loc_col2]=m2
        
        residuals = bin_resp_selected2- f(time_bin_indices_selected2, p2)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((bin_resp_selected2-np.mean(bin_resp_selected2))**2)
        r_squared = 1 - (ss_res / ss_tot)
        r2_3.append(r_squared)
    
runtimeN3=(time.time()-start_time_0)/60
print('polyfit finished')

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()
print('curvefit with log started')
tau_4=np.zeros((frame_size,frame_size))
r2_4=[]
for loc_row3 in range(frame_size):
    for loc_col3 in range(frame_size):
        bin_resp3=bin_array[spectral_index,:,loc_row3,loc_col3]
        time_index_max3=bin_resp3.argmax()
        time_bin_selected3=bin_size[time_index]-time_index_max3-1
        time_bin_indices_selected3=time_indices[:-time_bin_selected3]
        bin_resp_selected3=bin_resp3[:-time_bin_selected3]# Look out for the 2nd dimension
        bin_resp_selected3=np.squeeze(bin_resp_selected3)
        bin_resp_selected3=np.nan_to_num(np.log(bin_resp_selected3),posinf=0, neginf=0)
        popt1, pcov1 = curve_fit(lambda t, m, c: m * t + c, time_bin_indices_selected3, bin_resp_selected3)
        m3 = popt1[0]
        c3 = popt1[1]
        # p2 = np.polyfit(time_bin_indices_selected2, bin_resp_selected2, 1)
        # m2 = p2[0]
        # c2 = p2[1]
        tau_4[loc_row3,loc_col3]=m3
        
        residuals = bin_resp_selected3- f(time_bin_indices_selected3, popt1)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((bin_resp_selected3-np.mean(bin_resp_selected3))**2)
        r_squared = 1 - (ss_res / ss_tot)
        r2_4.append(r_squared)
    
runtimeN4=(time.time()-start_time_0)/60
print('curvefit with log finished')







# print(f'Values of m: {m1:5.3f}, {m2:5.3f}, {m3:5.3f}. Values of c: {c1:5.3f}, {c2:5.3f}, {c3:5.3f}')
# print(f'Values of m: {m3:5.3f}. Values of c:{c3:5.3f}')')
print('Runtime CF:%.4f LR:%.4f PF:%.4f CF_log:%.1f'%(runtimeN1,runtimeN2,runtimeN3,runtimeN4))
print('R_sq LR:%.4f CF:%.4f PF:%.4f CF_log:%.1f'%(np.mean(r2_1),np.mean(r2_2),np.mean(r2_3),np.mean(r2_4)))
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
plt.figure(11)
plt.imshow(tau_1,cmap='gray')
plt.colorbar()
plt.show()
plt.title('Linear Regression')

plt.figure(12)
plt.imshow(tau_2,cmap='gray')
plt.colorbar()
plt.show()
plt.title('Curvefit')

plt.figure(13)
plt.imshow(tau_3,cmap='gray')
plt.colorbar()
plt.show()
plt.title('Polyfit')

plt.figure(14)
plt.imshow(tau_4,cmap='gray')
plt.colorbar()
plt.show()
plt.title('Curvefit-linear')