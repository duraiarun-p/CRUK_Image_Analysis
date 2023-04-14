#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 20:33:26 2023

@author: Arun PDRA, THT
"""
#%%
import sys
sys.path.append("..")
from os import listdir
from os.path import join, isdir
from timeit import default_timer as timer

from flt_img_est_lib import flt_per_tile
from scipy.io import savemat


#%%
mypath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Normal/Row-1_Col-1_20230303'
save_path ='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/Test_Output'
tile_file=3
decimate_factor=3
spec_resampled=80
spec_truncated=300
# mypath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-1_20230214'
# List of responses of all tiles
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isdir(join(mypath, f))]
onlyfiles.sort()
onlyfiles_len=len(onlyfiles)

start_time_0_I=timer()

for tile_file in range(onlyfiles_len):
    

    matfile_list=listdir(onlyfiles[tile_file])
    #iterable tile
    matfile_list_path=join(onlyfiles[tile_file],matfile_list[0])#picking the mat file
    
    start_time_0=timer()
    img_int,img_flt,wave_spectrum,wave_spectrum_new=flt_per_tile(matfile_list_path,decimate_factor,spec_resampled,spec_truncated)
    runtimeN3=(timer()-start_time_0)/60
    mdic = {"img_int": img_int, "img_flt":img_flt, "wave_spectrum":wave_spectrum,"wave_spectrum_new":wave_spectrum_new,"runtimeN3":runtimeN3}
    print('Tile built time %s'%runtimeN3)
    matfile_filename=save_path+'/Row-1_Col-1_20230303-'+str(tile_file)+'.mat'
    savemat(matfile_filename, mdic)

runtimeN3=(timer()-start_time_0_I)/60
print('All Tiles built time %s'%runtimeN3)