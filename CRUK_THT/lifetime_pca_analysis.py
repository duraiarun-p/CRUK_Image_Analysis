#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 09:56:10 2023

@author: arun
"""

#%%


import h5py
import numpy as np
from timeit import default_timer as timer
from numpy.linalg import eigh
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# import mat73

from spectral import principal_components
from scipy.io import savemat

# import ipywidgets as ipyw

#%%
matfile='/home/arun/Documents/PyWSPrecision/CRUK_Image_Analysis/CRUK_THT/CRUK/Row_1_Col_1_N/workspace.frame_1.mat'

start_time_0=timer()
mat_contents=h5py.File(matfile,'r')
mat_contents_list=list(mat_contents.keys())
runtimeN0=(timer()-start_time_0)/60
print(runtimeN0)

start_time_0=timer()
bin_array_ref=mat_contents['bins_array_3']
frame_size_x_ref=mat_contents['frame_size_x']
hist_mode_ref=mat_contents['HIST_MODE']
binWidth_ref=mat_contents['binWidth']
runtimeN0=(timer()-start_time_0)/60
print(runtimeN0)

start_time_0=timer()
bin_array0=bin_array_ref[()]
frame_size=int(frame_size_x_ref[()])
hist_mode=int(hist_mode_ref[()])
binWidth=float(binWidth_ref[()])# time in ns
runtimeN0=(timer()-start_time_0)/60
print(runtimeN0)




#%%


time_interval=binWidth

time_resolution=(binWidth*1000)/(2*2**hist_mode)# time unit in ps

bin_size=np.shape(bin_array0)
time_index=2

time_indices=np.arange(bin_size[time_index])
time_line=time_indices*time_interval# Time axis for fitting data

spectral_index=100 #stride over spectral dimension

spectral_span_sum=32

spectra_len=bin_size[-1]
#%%
pc_comp_num=100
time_sbin=14
img = bin_array0[:,:,time_sbin,:]
# pca=PCA(n_components=pc_comp_num)
# img_pc=pca.fit_transform(img)
# ev=pca.explained_variance_ratio_
# pc_comp_num=20
# pc_0999 = pc.reduce(fraction=0.85)
def dim_red(imgx,pc_comp_num):
    pcx = principal_components(imgx)
    pc_0999x = pcx.reduce(num=pc_comp_num)
    img_pcx = pc_0999x.transform(imgx)
    return img_pcx
    
pc = principal_components(img)
pc_0999 = pc.reduce(num=pc_comp_num)
img_pc = pc_0999.transform(img)
v=pc.cov
# v = np.cov(img_pc, rowvar=False)

egnvalues, egnvectors = pc.eigenvalues, pc.eigenvectors
#
# Determine explained variance
#
total_egnvalues = sum(egnvalues)
var_exp = [(i/total_egnvalues) for i in sorted(egnvalues, reverse=True)]

# ev=pc.explained_variance_ratio_
#%%
plt.figure(1)
plt.plot(np.cumsum(var_exp))
plt.xlabel('PCs index')
plt.ylabel('Cum Expl.Variance')
#%%
img_pc1=dim_red(img,pc_comp_num)
mdic={'img_pc1':img_pc1}
savemat('flt_pca.mat', mdic)