#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 20:33:26 2023

@author: Arun PDRA, THT
"""
#%%

import sys
sys.path.append("..")
import resource
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

import os
from os import listdir
from os.path import join, isdir
from timeit import default_timer as timer

from flt_img_est_lib import flt_per_tile
from scipy.io import savemat


#%%
# mypath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Normal/Row-1_Col-1_20230303'
# mypath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-1_20230214'
# mypath ='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-6_Col-10_20230223'
mypath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-2_Col-3_20230216'

# mypath ='/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/scs/groups/kdhaliwa-CRUKEDD/TMA/FS-FLIM/raw/Normal_1B/Row-1_Col-1_20230303'

mypath_splitted = mypath.replace('/',',')
mypath_splitted = mypath_splitted.split(',')
tissue_core_file_name=mypath_splitted[-1]
save_path ='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/Test_Output'+'/'+tissue_core_file_name
if not os.path.exists(save_path):
    os.makedirs(save_path)
# Make a new directory for each tissue core mkdir tissue_core_file_name

tile_file=-1
decimate_factor=15
spec_resampled=20
spec_truncated=330
# mypath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-1_20230214'
# List of responses of all tiles
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isdir(join(mypath, f)) and f.startswith('R')]
onlyfiles.sort()
onlyfiles_len=len(onlyfiles)

start_time_0_I=timer()
onlyfiles_len=1

# for tile_file in range(onlyfiles_len):
    

#     matfile_list=listdir(onlyfiles[tile_file])
#     #iterable tile
#     matfile_list_path=join(onlyfiles[tile_file],matfile_list[0])#picking the mat file
    
#     start_time_0=timer()
#     img_int,img_flt,wave_spectrum,wave_spectrum_new,wave_spectrum_new_select=flt_per_tile(matfile_list_path,decimate_factor,spec_resampled,spec_truncated)
#     runtimeN3=(timer()-start_time_0)/60
#     mdic = {"img_int": img_int, "img_flt":img_flt, "wave_spectrum":wave_spectrum,"wave_spectrum_new":wave_spectrum_new,"runtimeN3":runtimeN3}
#     # mdic ={"tile_file":tile_file}
#     print('Tile built time %s'%runtimeN3)
#     matfile_filename=save_path+'/'+tissue_core_file_name+'-'+str(tile_file)+'.mat'
#     print(matfile_filename)
#     savemat(matfile_filename, mdic)
    
matfile_list=listdir(onlyfiles[tile_file])
#iterable tile
matfile_list_path=join(onlyfiles[tile_file],matfile_list[0])#picking the mat file

start_time_0=timer()
img_int,img_flt,wave_spectrum,wave_spectrum_new,wave_spectrum_new_select=flt_per_tile(matfile_list_path,decimate_factor,spec_resampled,spec_truncated)
runtimeN3=(timer()-start_time_0)/60
mdic = {"img_int": img_int, "img_flt":img_flt, "wave_spectrum":wave_spectrum,"wave_spectrum_new":wave_spectrum_new,"runtimeN3":runtimeN3}
# mdic ={"tile_file":tile_file}
print('Tile built time %s'%runtimeN3)
matfile_filename=save_path+'/'+tissue_core_file_name+'-'+str(tile_file)+'.mat'
print(matfile_filename)
savemat(matfile_filename,mdic)

runtimeN3=(timer()-start_time_0_I)/60
print('All Tiles built time %s'%runtimeN3)
#%%
img_flt[img_flt>5]=0
from matplotlib import pyplot as plt
plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(img_int[:,:,-1], cmap='gray')
plt.colorbar()
plt.show()
plt.subplot(1,2,2)
plt.imshow(img_flt[:,:,-1], cmap='gray')
plt.colorbar()
plt.show()