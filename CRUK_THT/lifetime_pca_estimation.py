#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 19:52:58 2023

@author: arun
"""
#%%


import h5py
import numpy as np
from timeit import default_timer as timer

from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA

# import mat73

from spectral import principal_components

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

# X=bin_array0[256,256,14,:]
# X1=[X,X]

# X2=np.zeros((spectra_len,spectra_len))

# for spec in range(spectra_len):
#     X2[:,spec]=X

# #%%
# plt.figure(21)
# plt.plot(X)


# #%%
# pca = PCA(n_components=20)
# pca_cmp=pca.fit(X2).transform(X2)

#%%

img = bin_array0[:,:,14,:]
pc = principal_components(img)

v = pc.cov

plt.figure(21)
plt.imshow(img[:,:,25],cmap='gray')
plt.show()

plt.figure(22)
plt.imshow(v,cmap='gray')
plt.show()

#%%
pc_0999 = pc.reduce(fraction=0.85)
img_pc = pc_0999.transform(img)

plt.figure(23)
plt.imshow(img_pc[:,:,3],cmap='gray')
plt.colorbar()
plt.show()
#%%

# bin_array0_1=np.zeros_like(bin_array0)

bin_array0_1=np.zeros((bin_size[0],bin_size[1],bin_size[time_index],2))

start_time_0=timer()

for time in range(bin_size[time_index]):
    img1 = bin_array0[:,:,time,:]
    pc = principal_components(img1)
    pc_0999 = pc.reduce(fraction=0.85)
    img_pc1 = pc_0999.transform(img1)
    pc_size=np.shape(img_pc1)
    print(time)
    if len(pc_size)<3:
        img_pc11=img_pc1
        pc_spectra=1
        for pix_x in range(bin_size[0]):
            for pix_y in range(bin_size[1]):
                bin_array0_1[pix_x,pix_y,time,pc_spectra]=img_pc11[pix_x,pix_y]
    else:
        pc_spectra=1
        img_pc12=img_pc1[:,:,0]
        for pix_x in range(bin_size[0]):
            for pix_y in range(bin_size[1]):
                # bin_array0_1[pix_x,pix_y,time,pc_spectra]=img_pc12[pix_x,pix_y,pc_spectra]
                bin_array0_1[pix_x,pix_y,time,pc_spectra]=img_pc12[pix_x,pix_y]
    
            
    # bin_array0_1[:,:,time,:]=img_pc

runtimeN0=(timer()-start_time_0)/60
print(runtimeN0)

#%%
loc_row1=150
loc_col1=256
bin_resp=bin_array0_1[loc_row1,loc_col1,:,1]

plt.figure(24)
# plt.imshow(img_pc[:,:,3],cmap='gray')
plt.plot(bin_resp)
# plt.colorbar()
plt.show()

#%%
#%%
from lifetime_estimate_lib import life_time_image_reconstruct_1_concurrent
import multiprocessing
from scipy import signal as sig

n_cores=multiprocessing.cpu_count()
bin_array=bin_array0_1[:,:,:,1]

bin_list=[]
bin_log_list=[]
bin_log_list_partial=[]
bin_index_list=[]
time_list=[]
time_list_partial=[]

count=0

start_time_0=timer()

for loc_row1 in range(frame_size):
    for loc_col1 in range(frame_size):
        # bin_resp=bin_array[spectral_index,:,loc_row1,loc_col1]
        # bin_resp=np.squeeze(bin_array[loc_row1,loc_col1,:,spectral_index])
        # bin_resp=bin_array[loc_row1,loc_col1,:,spectral_index]
        bin_resp=bin_array[loc_row1,loc_col1,:]
        # time_index_max=bin_resp.argmax()
        time_index_max=np.max(np.where(bin_resp==max(bin_resp)))
        count=count+1
        # if count == 643:
        #     print(count)
        #     pdb.set_trace()
            # time_index_max[time_index_max<8]=14 # Caused by low photon count
        # if time_index_max<14:
        #     time_index_max=14
        time_bin_selected=bin_size[time_index]-time_index_max-1
        # if time_bin_selected==0:
        #     time_bin_selected=1
        time_bin_indices_selected=time_indices[:-time_bin_selected]
        time_line_selected=time_line[time_bin_indices_selected]# x data for fitting
        bin_resp_selected=bin_resp[:-time_bin_selected]# Look out for the 2nd dimension
        bin_resp_selected=np.squeeze(bin_resp_selected)# y data for fitting
        bin_resp_selected=np.flip(bin_resp_selected)# Flipped for the real decay phenomenon

        bin_resp_selected_log=np.nan_to_num(np.log(bin_resp_selected),posinf=0, neginf=0) # log(y) data for fitting
        
        bin_index_list.append([loc_row1,loc_col1])
        bin_list.append(bin_resp_selected)
        # bin_log_list.append(np.nan_to_num(np.log(bin_resp_selected),posinf=0, neginf=0))
        time_list.append(time_line_selected)
        time_list_partial.append(time_line_selected[:4])
        # bin_log_list_partial.append(bin_resp_selected_log[:4])
        
runtimeN0=(timer()-start_time_0)/60
        
bin_Len=len(bin_list) # total number of pixel elements

#%%



bin_Len=len(bin_list) # total number of pixel elements

start_time_0=timer()
# tau_1_array,r_1=life_time_image_reconstruct_1(frame_size,bin_Len,bin_list,time_line,bin_index_list)
tau_1_array,r_1=life_time_image_reconstruct_1_concurrent(frame_size,bin_Len,bin_list,time_list,bin_index_list,n_cores)
tau_1_array[tau_1_array>np.median(tau_1_array)*20]=0 # For visualisation
tau_1_array=sig.medfilt2d(tau_1_array)
# tau_1_array1 = ma.masked_array(tau_1_array, bin_int_array_mask)
runtimeN1=(timer()-start_time_0)/60

# runtimeN_F=(timer()-start_time_0F)/60

#%%
plt.figure(29)
# plt.subplot(121)
# plt.imshow(tau_1_array,cmap='gray')
# plt.colorbar()
# plt.subplot(122)
plt.imshow(tau_1_array,cmap='gray')
plt.colorbar()
plt.show()
plt.title('Curvefit-Exp fitting with $R^2$:%.3f'%r_1)