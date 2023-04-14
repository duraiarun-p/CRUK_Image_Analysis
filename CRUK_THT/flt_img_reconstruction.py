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

#%%
mypath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Normal/Row-1_Col-1_20230303'
tile_file=3
decimate_factor=3
spec_resampled=100
# mypath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-1_20230214'
# List of responses of all tiles
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isdir(join(mypath, f))]
onlyfiles.sort()

# Mat file per tile
tile_file=3
start_time_0=timer()
img_int,img_flt,wave_spectrum_new,wave_spectrum_new_resampled=flt_per_tile(mypath,tile_file,decimate_factor,spec_resampled)
runtimeN3=(timer()-start_time_0)/60
print('Tile built time %s'%runtimeN3)