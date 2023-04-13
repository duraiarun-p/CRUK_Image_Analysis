#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 20:33:26 2023

@author: Arun PDRA, THT
"""
#%%
from os import listdir
from os.path import join, isdir

from fit_est_lib.py import flt_per_tile

#%%
mypath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Normal/Row-1_Col-1_20230303'
tile_file=3
decimate_factor=5
spec_resampled=75
# mypath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-1_20230214'
# List of responses of all tiles
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isdir(join(mypath, f))]
onlyfiles.sort()

# Mat file per tile
tile_file=3

img_int,img_flt=flt_per_tile(mypath,tile_file,decimate_factor,spec_resampled)