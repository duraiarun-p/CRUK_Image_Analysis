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

# bin_array0=bin_array0-bin_mean

spectral_span_sum=8
#%%
spectral_index=100

bin_array=np.sum(bin_array0[:,:,:,spectral_index:spectral_index+spectral_span_sum],-1)
#%%
# Similar to movsum
# bin_array_0=np.zeros_like(bin_array)
# for cum_index in range(bin_size[3]):
#     spectral_span=np.min([bin_size[3]-cum_index,spectral_span_sum]) 
#     print(spectral_span)
#     bin_array_slice=bin_array[:,:,:,cum_index:spectral_span]
#     bin_array_0[:,:,:,cum_index]=np.sum(bin_array_slice,axis=-1)
    
# bin_array=bin_array_0
# del bin_array_0

#%% Array Slicing based on spectrum/wavelength and parameter selection
time_interval=binWidth

time_resolution=(binWidth*1000)/(2*2**hist_mode)# time unit in ps

# bin_size=np.shape(bin_array)
time_index=2

time_indices=np.arange(bin_size[time_index])
time_line=time_indices*time_interval# Time axis for fitting data



#%% Array slicing based on fluorecence decay (photon count)
loc_row=np.random.randint(frame_size-1, size=(1))
loc_col=np.random.randint(frame_size-1, size=(1))

# bin_resp=bin_array[spectral_index,:,loc_row,loc_col]
# time_index_max=bin_resp.argmax()
# bin_resp=np.squeeze(bin_array[loc_row,loc_col,:,spectral_index])
bin_resp=np.squeeze(bin_array[loc_row,loc_col,:])
time_index_max=np.max(np.where(bin_resp==max(bin_resp)))
time_bin_selected=bin_size[time_index]-time_index_max-1
time_bin_indices_selected=time_indices[:-time_bin_selected]
time_line_selected=time_line[time_bin_indices_selected]# x data for fitting
bin_resp_selected=bin_resp[:-time_bin_selected]# Look out for the 2nd dimension
bin_resp_selected=np.squeeze(bin_resp_selected)# y data for fitting
bin_resp_selected=np.flip(bin_resp_selected)# Flipped for the real decay phenomenon

bin_resp_selected_log=np.nan_to_num(np.log(bin_resp_selected),posinf=0, neginf=0) # log(y) data for fitting

#%% Raw Data Visualisation


# plt.figure(100)
fig, ax1 = plt.subplots()
# ax1.plot(time_line_selected,bin_resp_selected)
ax1.scatter(time_line_selected, bin_resp_selected, c='b', marker='o', edgecolors='b', s=10)
ax1.set_ylabel(r'Intensity (counts)', color='b')
ax1.tick_params(axis='y', color='b', labelcolor='b')
ax1.set_title('Fluorescence decay - linearised and non-linearised')
ax2 = ax1.twinx()
# ax2.plot(time_line_selected,bin_resp_selected_log,'C1')
ax2.scatter(time_line_selected, bin_resp_selected_log, c='r', marker='o', edgecolors='r', s=10)
ax2.set_ylabel(r'Intensity (log(counts))', color='r')
ax2.tick_params(axis='y', color='r', labelcolor='r')
ax2.spines['right'].set_color('r')
ax2.spines['left'].set_color('b')
ax1.set_xlabel(r'time (ns)')

ax1.set_xlim(-0.1)
ax1.set_ylim(-0.1)
ax2.set_xlim(-0.1)
ax2.set_ylim(-0.1)

fig.legend(['Exp','Linear'], loc='upper left',bbox_to_anchor=(0.7, 0.85))

plt.show()
# plt.savefig('0.png')

#%% Curve fitting analysis

def f(t,p):
    m=p[0]
    c=p[1]
    return  (m * t) + c

def f1(t,popt):
    a = popt[0]
    b = popt[1]
    c = popt[2]
    return a * np.exp(b * -t) - c

def plotmyfit_1(fignum,x,y,popt,r_squared,labelstring,filename):
    a = popt[0]
    b = popt[1]
    c = popt[2]
    plt.figure(fignum)
    # x=time_bin_indices_selected
    # y=bin_resp_selected
    ax = plt.axes()
    ax.scatter(x, y, c='gray', marker='o', edgecolors='k', s=18, label='Raw data')
    # xlim = np.array(ax.get_xlim())
    # xlim[0] = 0
    # ax.plot(xlim, 2 * xlim + 11, 'k--', label='True underlying relationship')
    # ax.plot(x, m2 * xlim + c2, 'b', label='polyfit tool')
    ax.plot(x, f1(x,popt), 'k', label=labelstring)
    ax.set_title(r'Fluorecence lifetime estimate $\tau$=%.4f (R score: %.4f)'%(abs(b),r_squared))
    ax.set_xlabel(r'time (ns)')
    ax.set_ylabel(r'Intensity (counts)')
    ax.set_xlim(-0.1)
    ax.set_ylim(-0.1)
    ax.legend(fontsize=8)
    plt.show()
    plt.savefig(filename)
    
    
def plotmyfit_2(fignum,x,y,p,r_squared,labelstring,ylabelstring,filename):
    m=p[0]
    c=p[1]
    plt.figure(fignum)
    # x=time_bin_indices_selected
    # y=bin_resp_selected
    ax = plt.axes()
    ax.scatter(x, y, c='gray', marker='o', edgecolors='k', s=18, label='Raw data')
    # xlim = np.array(ax.get_xlim())
    # xlim[0] = 0
    # ax.plot(xlim, 2 * xlim + 11, 'k--', label='True underlying relationship')
    # ax.plot(x, m2 * xlim + c2, 'b', label='polyfit tool')
    ax.plot(x, f(x,p), 'k', label=labelstring)
    ax.set_title(r'Fluorecence lifetime estimate $\tau$=%.4f (R score: %.4f)'%(abs(m),r_squared))
    ax.set_xlabel(r'time (ns)')
    ax.set_ylabel(ylabelstring)
    ax.set_xlim(-0.1)
    ax.set_ylim(-0.1)
    ax.legend(fontsize=8)
    plt.show()
    # plt.savefig(filename)
    
def flt_est_cf_exp(f1,time_line_selected, bin_resp_selected):
    popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(b * -t) - c, time_line_selected, bin_resp_selected,maxfev=50000)
    # popt, pcov = curve_fit(f1, time_bin_indices_selected, bin_resp_selected)
    a = popt[0]
    b = popt[1]
    c = popt[2]
    residuals = bin_resp_selected- f1(time_line_selected, popt)
    # residuals = bin_resp_selected - (a * np.exp(b * time_bin_indices_selected) + c)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((bin_resp_selected-np.mean(bin_resp_selected))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return popt, r_squared

def flt_est_linreg_ls(time_line_selected, bin_resp_selected_log):
    p1 = stats.linregress(time_line_selected, bin_resp_selected_log)
    m1 = p1[0]
    c1 = p1[1]
    residuals1 = bin_resp_selected_log- f(time_line_selected, p1)
    ss_res1 = np.sum(residuals1**2)
    ss_tot1 = np.sum((bin_resp_selected_log-np.mean(bin_resp_selected_log))**2)
    r_squared1 = 1 - (ss_res1 / ss_tot1)
    return p1, r_squared1

def flt_est_polyfit_ls(time_line_selected, bin_resp_selected_log):
    p2 = np.polyfit(time_line_selected, bin_resp_selected_log, 1)
    m2 = p2[0]
    c2 = p2[1]
    residuals2 = bin_resp_selected_log- f(time_line_selected, p2)
    ss_res2 = np.sum(residuals2**2)
    ss_tot2 = np.sum((bin_resp_selected_log-np.mean(bin_resp_selected_log))**2)
    r_squared2 = 1 - (ss_res2 / ss_tot2)
    return p2, r_squared2

def flt_est_cf_ls(time_line_selected, bin_resp_selected_log):
    popt1, pcov1 = curve_fit(lambda t, m, c: m * t + c, time_line_selected, bin_resp_selected_log,maxfev=50000)
    m3 = popt1[0]
    c3 = popt1[1]
    residuals3 = bin_resp_selected_log- f(time_line_selected, popt1)
    ss_res3 = np.sum(residuals3**2)
    ss_tot3 = np.sum((bin_resp_selected_log-np.mean(bin_resp_selected_log))**2)
    r_squared3 = 1 - (ss_res3 / ss_tot3)
    return popt1, r_squared3

def flt_est_cf_ls_log(time_line_selected, bin_resp_selected):
    popt1, pcov1 = curve_fit(lambda t, m, c: m * t + c, time_line_selected, bin_resp_selected,maxfev=50000)
    m3 = popt1[0]
    c3 = popt1[1]
    residuals3 = bin_resp_selected- f(time_line_selected, popt1)
    ss_res3 = np.sum(residuals3**2)
    ss_tot3 = np.sum((bin_resp_selected_log-np.mean(bin_resp_selected))**2)
    r_squared3 = 1 - (ss_res3 / ss_tot3)
    return popt1, r_squared3
    
# plotmyfit(100,time_line_selected,bin_resp_selected)
# plotmyfit(101,time_line_selected,bin_resp_selected_log)

popt1, r_squared1=flt_est_cf_exp(f1,time_line_selected, bin_resp_selected)
plotmyfit_1(100,time_line_selected,bin_resp_selected,popt1, r_squared1,'Exp using Curve fit','1.png')

p2, r_squared2=flt_est_linreg_ls(time_line_selected, bin_resp_selected_log)
plotmyfit_2(101,time_line_selected,bin_resp_selected_log,p2, r_squared2,'Linear Regression','Intensity (log (counts))','2.png')

p3, r_squared3=flt_est_polyfit_ls(time_line_selected, bin_resp_selected_log)
plotmyfit_2(102,time_line_selected,bin_resp_selected_log,p3, r_squared3,'Polyfit','Intensity (log (counts))','3.png')

p4, r_squared4=flt_est_cf_ls(time_line_selected, bin_resp_selected_log)
plotmyfit_2(103,time_line_selected,bin_resp_selected_log,p4, r_squared4,'LS using Curve fit','Intensity (log (counts))','4.png')

p5, r_squared5=flt_est_cf_ls_log(time_line_selected[:4], bin_resp_selected[:4])
plotmyfit_2(104,time_line_selected[:4],bin_resp_selected[:4],p5, r_squared5,'LS using Curve fit (Exp Partial Decay)','Intensity (log (counts))','5.png')

p6, r_squared6=flt_est_cf_ls(time_line_selected[:4], bin_resp_selected_log[:4])
plotmyfit_2(105,time_line_selected[:4],bin_resp_selected_log[:4],p6, r_squared6,'LS using Curve fit (Linear Partial Decay)','Intensity (log (counts))','6.png')
#%%


#%%

plt.figure(200)
x=time_line_selected
y=bin_resp_selected_log
ax = plt.axes()
ax.scatter(x, y, c='gray', marker='o', edgecolors='k', s=18, label='raw data')
xlim = np.array(ax.get_xlim())
xlim[0] = 0
ax.plot(x, p2[0] * x + p2[1], 'b-x', label=['LR R^2 = ' + str(round(r_squared2,3))])
ax.plot(x, p3[0] * x + p3[1], 'g-x', label=['PF R^2 = ' + str(round(r_squared3,3))])
ax.plot(x, p4[0] * x + p4[1], 'r-x', label=['CF R^2 = ' + str(round(r_squared4,3))])

ax.set_title(r'Fluorecence lifetime estimated as $\tau$=%.4f using fitting techniques'%(abs(p2[0])))
ax.set_xlabel(r'time (ns)')
ax.set_ylabel(r'Intensity log(counts)')
ax.set_xlim(xlim)
ax.set_ylim(0)
ax.legend(fontsize=8)
plt.show()
# plt.savefig('7.png')