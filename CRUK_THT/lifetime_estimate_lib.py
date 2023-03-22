#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 00:51:50 2023

@author: arun
"""
#%%
import sys
import os

from matplotlib import pyplot as plt
from scipy.io import savemat
import h5py

import numpy as np
import random
from scipy.optimize import curve_fit
from scipy import stats

# from multiprocessing import Process
# from multiprocessing import Pool
import multiprocessing
from numba import jit
from numba import cuda

import concurrent.futures
from traceback import print_exc

from timeit import default_timer as timer 




#%%

def f(t,p):
    m=p[0]
    c=p[1]
    return  (m * t) + c

def f1(t,popt):
    a = popt[0]
    b = popt[1]
    c = popt[2]
    return a * np.exp(b * -t) - c

def flt_est_cf_exp(bin_resp_selected,time_line_selected):
    # popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(b * -t) - c, bin_resp_selected, time_line_selected,maxfev=50000)
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
    return abs(b), r_squared

def life_time_est_cf_exp(bin_resp,time_line):
    time_index_max=bin_resp.argmax()
    time_bin_selected=bin_size[time_index]-time_index_max-1
    time_bin_indices_selected=time_indices[:-time_bin_selected]
    time_line_selected=time_line[time_bin_indices_selected]# x data for fitting
    bin_resp_selected=bin_resp[:-time_bin_selected]# Look out for the 2nd dimension
    bin_resp_selected=np.squeeze(bin_resp_selected)# y data for fitting
    bin_resp_selected=np.flip(bin_resp_selected)# Flipped for the real decay phenomenon
    popt, r_squared=flt_est_cf_exp(time_line_selected, bin_resp_selected)
    tau=abs(popt[1])
    return tau, r_squared

def life_time_image_reconstruct_1(frame_size,bin_Len,bin_list,time_line):
    tau_1_array=np.zeros((frame_size,frame_size))    
    tau_1=np.zeros((bin_Len,1))
    r_1=np.zeros((bin_Len,1))
    for bin_index in range(bin_Len):
        bin_resp=bin_list[bin_index]
        tau, r_squared=life_time_est_cf_exp(bin_resp,time_line)
        tau_1[bin_index]=tau
        r_1[bin_index]=r_squared
        tau_row=bin_index_list[bin_index][0]
        tau_col=bin_index_list[bin_index][1]
        tau_1_array[tau_row,tau_col]=tau
    tau_1_array[tau_1_array>(np.mean(tau_1_array)*10)]=0
    return tau_1_array

def life_time_image_reconstruct_1_concurrent(frame_size,bin_Len,bin_list,time_list,bin_index_list,n_cores):
    tau_1_array=np.zeros((frame_size,frame_size))    
    tau_1=np.zeros((bin_Len,1))
    r_1=np.zeros((bin_Len,1))
    res=[]
    resparlist=[]
    reslist=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:       
        futures = {executor.submit(flt_est_cf_exp,bin_resp,time_line) for bin_resp,time_line in zip(bin_list,time_list)}
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            resparlist.append(result)
        for bin_index in range(bin_Len):
            tau_row=bin_index_list[bin_index][0]
            tau_col=bin_index_list[bin_index][1]
            tau_1[bin_index]=resparlist[bin_index][0]
            r_1[bin_index]=resparlist[bin_index][1]
            tau_1_array[tau_row,tau_col]=resparlist[bin_index][0]
    # tau_1_array[tau_1_array>(np.mean(tau_1_array)*10)]=0
    return tau_1_array,np.mean(r_1)

def life_time_image_reconstruct_1_concurrent_1(frame_size,bin_Len,bin_list,time_list,bin_index_list):
    tau_1_array=np.zeros((frame_size,frame_size))    
    tau_1=np.zeros((bin_Len,1))
    r_1=np.zeros((bin_Len,1))
    res=[]
    resparlist=[]
    reslist=[]
    # with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:       
    #     futures = {executor.submit(flt_est_cf_exp,bin_resp,time_line) for bin_resp,time_line in zip(bin_list,time_list)}
    #     for fut in concurrent.futures.as_completed(futures):
    #         result = fut.result()
    #         resparlist.append(result)
    for bin_index in range(bin_Len):
        # print(bin_index)
        bin_resp_x=bin_list[bin_index]
        time_line_x=time_list[bin_index]
        tau_row=bin_index_list[bin_index][0]
        tau_col=bin_index_list[bin_index][1]
        tau, r_squared=flt_est_cf_exp(bin_resp_x,time_line_x)
        # tau_1_1[loc_row,loc_col]=taus
        tau_1[bin_index]=tau
        r_1[bin_index]=r_squared
        tau_1_array[tau_row,tau_col]=tau
    tau_1_array[tau_1_array>(np.mean(tau_1_array)*10)]=0
    return tau_1_array,np.mean(r_1)

@jit(nogil=True)
def life_time_est_cf_exp_jit(bin_resp,time_line):
    time_index_max=bin_resp.argmax()
    time_bin_selected=bin_size[time_index]-time_index_max-1
    time_bin_indices_selected=time_indices[:-time_bin_selected]
    time_line_selected=time_line[time_bin_indices_selected]# x data for fitting
    bin_resp_selected=bin_resp[:-time_bin_selected]# Look out for the 2nd dimension
    bin_resp_selected=np.squeeze(bin_resp_selected)# y data for fitting
    bin_resp_selected=np.flip(bin_resp_selected)# Flipped for the real decay phenomenon
    popt, r_squared=flt_est_cf_exp(f1,time_line_selected, bin_resp_selected)
    tau=abs(popt[1])
    return tau, r_squared

# @jit(target_backend='cuda',nogil=True,parallel=True)
@jit(target_backend='cuda')
# @cuda.autojit 
def life_time_image_reconstruct_1_gpu(frame_size,bin_Len,bin_list,time_list):
    # tau_1_array=np.zeros((frame_size,frame_size))    
    tau_1=np.zeros((bin_Len,1))
    r_1=np.zeros((bin_Len,1))
    for bin_index in range(bin_Len):
        bin_resp=bin_list[bin_index]
        time_line=time_list[bin_index]
        tau, r_squared=flt_est_cf_exp(bin_resp,time_line)
        tau_1[bin_index]=tau
        r_1[bin_index]=r_squared
        # tau_row=bin_index_list[bin_index][0]
        # tau_col=bin_index_list[bin_index][1]
        # tau_1_array[tau_row,tau_col]=tau
    # tau_1_array[tau_1_array>(np.mean(tau_1_array)*10)]=0
    return {tau_1,r_1}

def flt_est_linreg_ls(time_line_selected, bin_resp_selected_log):
    p1 = stats.linregress(time_line_selected, bin_resp_selected_log)
    m1 = p1[0]
    c1 = p1[1]
    residuals1 = bin_resp_selected_log- f(time_line_selected, p1)
    ss_res1 = np.sum(residuals1**2)
    ss_tot1 = np.sum((bin_resp_selected_log-np.mean(bin_resp_selected_log))**2)
    r_squared1 = 1 - (ss_res1 / ss_tot1)
    return abs(m1), r_squared1

def life_time_image_reconstruct_2_concurrent(frame_size,bin_Len,bin_list,time_list,bin_index_list,n_cores):
    tau_1_array=np.zeros((frame_size,frame_size))    
    tau_1=np.zeros((bin_Len,1))
    r_1=np.zeros((bin_Len,1))
    res=[]
    resparlist=[]
    reslist=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:       
        futures = {executor.submit(flt_est_linreg_ls,time_line,bin_resp) for bin_resp,time_line in zip(bin_list,time_list)}
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            resparlist.append(result)
        for bin_index in range(bin_Len):
            tau_row=bin_index_list[bin_index][0]
            tau_col=bin_index_list[bin_index][1]
            tau_1[bin_index]=resparlist[bin_index][0]
            r_1[bin_index]=resparlist[bin_index][1]
            tau_1_array[tau_row,tau_col]=resparlist[bin_index][0]
    # tau_1_array[tau_1_array>(np.mean(tau_1_array)*10)]=0
    return tau_1_array,np.mean(r_1)

def flt_est_polyfit_ls(time_line_selected, bin_resp_selected_log):
    p2 = np.polyfit(time_line_selected, bin_resp_selected_log, 1)
    m2 = p2[0]
    c2 = p2[1]
    residuals2 = bin_resp_selected_log- f(time_line_selected, p2)
    ss_res2 = np.sum(residuals2**2)
    ss_tot2 = np.sum((bin_resp_selected_log-np.mean(bin_resp_selected_log))**2)
    r_squared2 = 1 - (ss_res2 / ss_tot2)
    return abs(m2), r_squared2

def life_time_image_reconstruct_3_concurrent(frame_size,bin_Len,bin_list,time_list,bin_index_list,n_cores):
    tau_1_array=np.zeros((frame_size,frame_size))    
    tau_1=np.zeros((bin_Len,1))
    r_1=np.zeros((bin_Len,1))
    res=[]
    resparlist=[]
    reslist=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:       
        futures = {executor.submit(flt_est_polyfit_ls,time_line,bin_resp) for bin_resp,time_line in zip(bin_list,time_list)}
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            resparlist.append(result)
        for bin_index in range(bin_Len):
            tau_row=bin_index_list[bin_index][0]
            tau_col=bin_index_list[bin_index][1]
            tau_1[bin_index]=resparlist[bin_index][0]
            r_1[bin_index]=resparlist[bin_index][1]
            tau_1_array[tau_row,tau_col]=resparlist[bin_index][0]
    tau_1_array[tau_1_array>(np.mean(tau_1_array)*10)]=0
    return tau_1_array,np.mean(r_1)

def flt_est_cf_ls(time_line_selected, bin_resp_selected_log):
    popt1, pcov1 = curve_fit(lambda t, m, c: m * t + c, time_line_selected, bin_resp_selected_log,maxfev=50000)
    m3 = popt1[0]
    c3 = popt1[1]
    residuals3 = bin_resp_selected_log- f(time_line_selected, popt1)
    ss_res3 = np.sum(residuals3**2)
    ss_tot3 = np.sum((bin_resp_selected_log-np.mean(bin_resp_selected_log))**2)
    r_squared3 = 1 - (ss_res3 / ss_tot3)
    return abs(m3), r_squared3

def life_time_image_reconstruct_4_concurrent(frame_size,bin_Len,bin_list,time_list,bin_index_list,n_cores):
    tau_1_array=np.zeros((frame_size,frame_size))    
    tau_1=np.zeros((bin_Len,1))
    r_1=np.zeros((bin_Len,1))
    res=[]
    resparlist=[]
    reslist=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:       
        futures = {executor.submit(flt_est_cf_ls,time_line,bin_resp) for bin_resp,time_line in zip(bin_list,time_list)}
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            resparlist.append(result)
        for bin_index in range(bin_Len):
            tau_row=bin_index_list[bin_index][0]
            tau_col=bin_index_list[bin_index][1]
            tau_1[bin_index]=resparlist[bin_index][0]
            r_1[bin_index]=resparlist[bin_index][1]
            tau_1_array[tau_row,tau_col]=resparlist[bin_index][0]
    # tau_1_array[tau_1_array>(np.mean(tau_1_array)*10)]=0
    return tau_1_array,np.mean(r_1)