#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 12:09:42 2023

@author: Arun PDRA, THT
"""

from aicsimageio import AICSImage

from aicsimageio.readers import CziReader

from aicspylibczi import CziFile

from pathlib import Path

#%%
img = AICSImage("/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/tumour_2B_HE.czi") # selects the first scene found
img.metadata 
cx=img.physical_pixel_sizes.X
print(cx)

# img = CziReader("/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/tumour_2B_HE.czi") # selects the first scene found
# m=img.metadata

# pth = Path("/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/tumour_2B_HE.czi")

# czi = CziFile(pth)

# dimensions = czi.get_dims_shape()