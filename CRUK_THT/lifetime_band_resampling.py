#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:08:27 2023

@author: Arun PDRA, THT
"""

#%%
from os import listdir
from os.path import isfile, join, isdir

import h5py
import numpy as np
from timeit import default_timer as timer

from scipy import signal as sig

from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA

# import mat73

from scipy import ndimage as ndi

from spectral import BandResampler

from lifetime_estimate_lib_THT import life_time_image_reconstruct_1_concurrent_1,life_time_image_reconstruct_1_concurrent,life_time_image_reconstruct_4_concurrent,life_time_image_reconstruct_2_concurrent,life_time_image_reconstruct_3_concurrent

#%%
# matfile='/home/arun/Documents/PyWSPrecision/CRUK_Image_Analysis/CRUK_THT/CRUK/Row_1_Col_1_N/workspace.frame_1.mat'
mypath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Normal/Row-1_Col-1_20230303'

# mypath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-1_20230214'
# List of responses of all tiles
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isdir(join(mypath, f))]
onlyfiles.sort()

# Mat file per tile
tile_file=3
matfile_list=listdir(onlyfiles[tile_file])
#iterable tile
matfile_list_path=join(onlyfiles[tile_file],matfile_list[0])#picking the mat file

start_time_0=timer()
mat_contents=h5py.File(matfile_list_path,'r+')
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

del bin_array_ref


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
loc_row=np.random.randint(frame_size-1, size=(1))
loc_col=np.random.randint(frame_size-1, size=(1))

# bin_resp=bin_array0[spectral_index,:,loc_row,loc_col]

#%%
bin_spec=bin_array0[loc_row,loc_col,14,:]
bin_spec=np.reshape(bin_spec, (spectra_len,1))
# bin_spec=np.cumsum(bin_spec)
window=11
# bin_spec_s=np.convolve(bin_spec, np.ones(window)/window, 'same')

bin_spec_s = sig.savgol_filter(bin_spec, window_length=window, polyorder=3, mode='constant',axis=0)

band_window=5# 1 component for every band_windowth component
bin_wave=np.arange(spectra_len)
bin_wave_new=np.arange(spectra_len//band_window)
resample_fn1 = BandResampler(bin_wave, bin_wave_new)

bin_spec_res = resample_fn1(bin_spec_s)

bin_spec_res_1=sig.resample(bin_spec_s, spectra_len//band_window,domain='time')

bin_spec_res_2=sig.resample(bin_spec_s, spectra_len//(band_window*2),domain='time')

bin_spec_res_3=sig.resample(bin_spec_s, spectra_len//(band_window*3),domain='time')

bin_spec_res_4=sig.resample(bin_spec_s, spectra_len//(band_window*4),domain='time')

plt.figure(1)
# plt.plot(bin_spec)
plt.plot(bin_spec_s)
plt.show()
plt.figure(2)
plt.plot(bin_spec_res)
plt.plot(bin_spec_res_1)
plt.plot(bin_spec_res_2)
plt.plot(bin_spec_res_3)
plt.plot(bin_spec_res_4)
plt.show()
##%%
# from spectres import spectres

# wave_spec=np.linspace(500,780,spectra_len)

# wave_spec_new_1=np.linspace(500,780,spectra_len//5)
# wave_spec_new_2=np.linspace(500,780,spectra_len//10)

# bin_spec=bin_array0[loc_row,loc_col,14,:]
# bin_spec=np.reshape(bin_spec, (spectra_len,1))
# bin_spec_res_1 = spectres(wave_spec_new_1, bin_spec, wave_spec)
# # bin_spec_res_2 = spectres(wave_spec_new_2, bin_spec_s, wave_spec)


# plt.figure(3)
# plt.plot(bin_spec_res_1)
# # plt.plot(bin_spec_res_2)
# plt.show()