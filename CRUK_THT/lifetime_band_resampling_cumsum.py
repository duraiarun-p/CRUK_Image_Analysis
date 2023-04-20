#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:13:47 2023

@author: Arun PDRA, THT
"""

#%%
from os import listdir
from os.path import isfile, join, isdir

import h5py
import numpy as np
from timeit import default_timer as timer

from scipy import signal as sig
from skimage import filters

from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA

# import mat73

# from spectral import BandResampler

from scipy import ndimage as ndi

import multiprocessing

from lifetime_estimate_lib_THT import life_time_image_reconstruct_1_concurrent,prepare_data_list
from flt_img_est_lib import resample_fn_2
#%%
def flt_img(bin_array2,spectral_index,time_index,fignum):
    
    bin_int=bin_array2[:,:,:,spectral_index]# only for spectral span = 1
    bin_array=bin_int
    
    # bin_int=bin_array[:,:,:]
    
    # bin_int_array_4_mask=bin_array[:,:,0]
    
    bin_int_array=np.sum(bin_int,axis=time_index)
    # bin_int_array=np.cumsum(bin_int,axis=time_index)
    # bin_int_array=bin_int_array[:,:,-1]
    
    bin_int_array1=filters.gaussian(bin_int_array, sigma=10)
    
    # Mean_Thresh=np.sqrt(np.mean(bin_int_array1))
    
    # thresh = threshold_otsu(bin_int_array1)
    # bin_int_array_mask=bin_int_array1>Mean_Thresh
    
    # Extract this threshold in the first file then carry over to the subsequent tiles
    #   thresh=(np.mean(bin_int_array)+np.min(bin_int_array1))*0.35
    bin_mask_thresh=np.mean(bin_int_array1)*.75
    bin_int_array_mask=np.zeros_like(bin_int_array1)
    bin_int_array_mask[bin_int_array1>bin_mask_thresh]=1
        # bin_int_array_mask = filters.sobel(bin_int_array_mask)
        # bin_int_array_mask=area_closing(bin_int_array_mask,area_threshold=64)
    bin_int_array_mask = ndi.binary_fill_holes(bin_int_array_mask)
    # bin_int_array_mask_edge = ndi.sobel(bin_int_array_mask)
    # struct2 = ndi.generate_binary_structure(3, 3)
    # bin_int_array_mask=ndi.binary_dilation(bin_int_array_mask, structure=struct2)
        # bin_int_array_mask = util.invert(bin_int_array_mask)
        # bin_int_array_mask= dilate(3, bin_int_array_mask)
        # bin_int_array_mask = ndi.binary_fill_holes(bin_int_array_mask)
        # bin_int_array_mask = np.invert(bin_int_array_mask)
        # bin_int_array_mask = ndi.binary_fill_holes(bin_int_array_mask)
        
    
    
    plt.figure(fignum+1)
    plt.imshow(bin_int_array,cmap='gray')
    plt.colorbar()
    plt.show()
    plt.title('Intensity')
    # plt.figure(27)
    # plt.imshow(bin_int_array_mask,cmap='gray')
    # plt.colorbar()
    # plt.show()
    # plt.title('Intensity_mask')
    # plt.figure(28)
    # plt.imshow(bin_int_array_mask_edge,cmap='gray')
    # plt.colorbar()
    # plt.show()
    # plt.title('Intensity_mask')
    ##%%
    
    
    
    
    
    
    ##%%
    
    
    
    
    n_cores=multiprocessing.cpu_count()
    
    bin_Len,bin_list,time_list,bin_index_list=prepare_data_list(frame_size,bin_array,bin_size,time_index,time_indices,time_line)
    
    start_time_0=timer()
    # tau_1_array=life_time_image_reconstruct_1(frame_size,bin_Len,bin_list,time_line,bin_index_list,n_cores)  
    tau_1_array,r_1=life_time_image_reconstruct_1_concurrent(frame_size,bin_Len,bin_list,time_list,bin_index_list,n_cores)
    # tau_1_array=sig.medfilt2d(tau_1_array)
    # if (np.max(tau_1_array)-np.min(tau_1_array))/np.mean(tau_1_array)>2:
    #     tau_1_array[tau_1_array>np.mean(tau_1_array)+((np.max(tau_1_array)-np.min(tau_1_array))/20)]=0
    tau_1_array[tau_1_array>np.median(tau_1_array)*50]=0 # For visualisation
    # tau_1_array=sig.medfilt2d(tau_1_array)
    tau_1_array1=sig.medfilt2d(tau_1_array)
    # tau_1_array1 = ma.masked_array(tau_1_array, bin_int_array_mask)
    # tau_1_array1 = np.multiply(tau_1_array, bin_int_array_mask)
    # tau_1,r_1=life_time_image_reconstruct_1_gpu(frame_size,bin_Len,bin_list,time_line)
    runtimeN1=(timer()-start_time_0)/60
    print(runtimeN1)
    
    ##%%
    plt.figure(fignum)
    # plt.subplot(121)
    # plt.imshow(tau_1_array,cmap='gray')
    # plt.colorbar()
    # plt.subplot(122)
    plt.imshow(tau_1_array1,cmap='gray')
    plt.colorbar()
    plt.show()
    plt.title('Curvefit-Exp fitting with $R^2$:%.3f'%r_1)

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

#%% Data Parameters and Specifications

time_interval=binWidth
time_resolution=(binWidth*1000)/(2*2**hist_mode)# time unit in ps

bin_size=np.shape(bin_array0)
time_index=2

time_indices=np.arange(bin_size[time_index])
time_line=time_indices*time_interval# Time axis for fitting data

spectral_index=100 #stride over spectral dimension

spectral_span_sum=16

spectra_len=bin_size[-1]
#%%
# bin_array=np.sum(bin_array0[:,:,:,spectral_index:spectral_index+spectral_span_sum],-1)

bin_array1=np.zeros_like(bin_array0)
#%% Moving sum over the spectral dimension
start_time_0=timer()
for spec_i in range(bin_size[-1]):
    spectral_span=np.min([bin_size[3]-spec_i,spectral_span_sum])
    # print(spectral_span)
    bin_array1[:,:,:,spec_i]=np.sum(bin_array0[:,:,:,spec_i:spec_i+spectral_span],-1)
    
runtimeN0=(timer()-start_time_0)/60
print(runtimeN0)

#%% 

loc_row=np.random.randint(frame_size-1, size=(1))
loc_col=np.random.randint(frame_size-1, size=(1))

bin_spec=bin_array1[loc_row,loc_col,14,:]
bin_spec=np.reshape(bin_spec, (spectra_len,1))


window=11
band_window=5# 1 component for every band_windowth component
bin_spec_s=bin_spec
# bin_spec_s = sig.savgol_filter(bin_spec, window_length=window, polyorder=3, mode='constant',axis=0)


bin_spec_res_1=sig.resample(bin_spec_s, spectra_len//band_window,domain='time')

bin_spec_res_2=sig.resample(bin_spec_s, spectra_len//(band_window*2),domain='time')

bin_spec_res_3=sig.resample(bin_spec_s, spectra_len//(band_window*3),domain='time')

bin_spec_res_4=sig.resample(bin_spec_s, spectra_len//(band_window*4),domain='time')

plt.figure(1)
# plt.plot(bin_spec)
plt.plot(bin_spec_s)
plt.show()
# plt.figure(2)
# plt.plot(bin_spec_res_1)
# plt.plot(bin_spec_res_2)
# plt.plot(bin_spec_res_3)
# plt.plot(bin_spec_res_4)
# plt.show()
#%% Spectral dimension reduction by resampling (Spectral resampling)

def resample_fn(bin_spec_x,decimate_factor_x):
    bin_spec_res_1_x=np.squeeze(sig.decimate(bin_spec_x,decimate_factor_x,ftype='fir'))
    return bin_spec_res_1_x


scale=1
decimate_factor=5
spec_len=spectra_len//(band_window*scale)


bin_spec=bin_array1[loc_row,loc_col,14,:330]
# bin_spec_res_1_bin=sig.decimate(bin_spec,decimate_factor,ftype='fir')

bin_spec_res_1_bin=resample_fn(bin_spec,decimate_factor)
spec_len_new=len(bin_spec_res_1_bin)

wave_spectrum=np.linspace(500, 780,bin_size[3])
# wave_samples_decimated=bin_size[3]//decimate_factor
wave_spectrum_new = np.linspace(500, 780, spec_len_new)

# bin_spec_res_1=np.squeeze(sig.decimate(bin_spec,3,ftype='fir'))

# bin_spec_res_2=sig.decimate(bin_spec,5,ftype='fir')

# bin_spec_res_3=sig.decimate(bin_spec,10,ftype='fir')

# bin_spec_res_4=sig.decimate(bin_spec,15,ftype='fir')
bin_spec=np.squeeze(bin_spec)

bin_spec_res_1=resample_fn_2(bin_spec,3)
bin_spec_res_2=resample_fn_2(bin_spec,5)
bin_spec_res_3=resample_fn_2(bin_spec,11)
bin_spec_res_4=resample_fn_2(bin_spec,15)


plt.figure(3)
# plt.plot(bin_spec)
plt.plot(bin_spec_s)
plt.legend('raw')
plt.show()
plt.figure(4)
plt.plot(bin_spec_res_1)
plt.plot(bin_spec_res_2)
plt.plot(bin_spec_res_3)
plt.plot(bin_spec_res_4)
plt.legend(['3','5','10','15'])
plt.show()

bin_spec_res_1=resample_fn(bin_spec,3)
bin_spec_res_2=resample_fn(bin_spec,5)
bin_spec_res_3=resample_fn(bin_spec,11)
bin_spec_res_4=resample_fn(bin_spec,15)
plt.figure(5)
plt.plot(bin_spec_res_1)
plt.plot(bin_spec_res_2)
plt.plot(bin_spec_res_3)
plt.plot(bin_spec_res_4)
plt.legend(['3','5','10','15'])
plt.show()

#%%

# spec_len=100
# start_spectrum=0
# # stop_spectrum=300
# stop_spectrum=bin_size[3]
# stop_spectrum_len=spec_len
# bin_array2=np.zeros((bin_size[0],bin_size[1],bin_size[2],spec_len_new))
# start_time_0=timer()
# for loc_row1 in range(bin_size[0]):
#     for loc_col1 in range(bin_size[1]):
#         for time_bin in range(bin_size[2]):
#             bin_spec1=bin_array1[loc_row1,loc_col1,time_bin,start_spectrum:stop_spectrum]
#             # bin_spec1=np.reshape(bin_spec1, (spectra_len,1))
#             # bin_spec_res_2_bin=sig.resample(bin_spec1, spectra_len//(band_window*scale),domain='time')
#             # bin_spec_res_2_bin=sig.decimate(bin_spec1,decimate_factor,ftype='fir')
#             bin_spec_res_2_bin=resample_fn(bin_spec1,decimate_factor)
#             bin_array2[loc_row1,loc_col1,time_bin,:]=bin_spec_res_2_bin

# runtimeN0=(timer()-start_time_0)/60
# print(runtimeN0)
#%%
spectral_index=5
# bin_array=np.sum(bin_array2[:,:,:,spectral_index],-1)
# bin_array_bck=np.mean(bin_array,2)# Selecting background based on time bin
# bin_mean=np.mean(bin_array)

# Background subtraction
# for time_bin in range(bin_size[time_index]):    
#     bin_array[:,:,time_bin]=bin_array[:,:,time_bin]-bin_array_bck

##%% An attempt to parallelize the loops

# def dilate(cycles, image):
#    image = Image.fromarray(image, 'L')
#    for _ in range(cycles):
#        image = image.filter(ImageFilter.MaxFilter(3))
#    return np.array(image)
# start_time_0=timer()
# flt_img(bin_array2,0,time_index,10)
# flt_img(bin_array2,24,time_index,20)
# flt_img(bin_array2,50,time_index,30)
# flt_img(bin_array2,55,time_index,40)
# flt_img(bin_array2,75,time_index,50)

# flt_img(bin_array2,90,time_index,70)

# flt_img(bin_array2,104,time_index,90)

# flt_img(bin_array2,115,time_index,100)

# flt_img(bin_array2,125,time_index,115)

# runtimeN0=(timer()-start_time_0)/60
# print(runtimeN0)
#%%
# bin_resp=bin_array2[255,255,:,101]