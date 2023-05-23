#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:37:39 2023

@author: Arun PDRA, THT
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
import resource
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

import os
from os import listdir
from os.path import join, isdir
from timeit import default_timer as timer
from scipy import ndimage as ndi


#%%

mypath='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-5_Col-11_20230224/FLT_IMG_DIR_4'
# mypath='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-9_20230222/FLT_IMG_DIR'

# onlyfiles = [join(mypath, f) for f in listdir(mypath) if isdir(join(mypath, f))]
onlyfiles = [join(mypath, f) for f in listdir(mypath)]
onlyfiles.sort()
#del onlyfiles[-1]# remove cube file from list
onlyfiles_len=len(onlyfiles)

flt_cube_list=[]
int_cube_list=[]
int_list=[]


for tile_file in range(onlyfiles_len):
    mat_fname=onlyfiles[tile_file]
    print(mat_fname)
    mat_contents = sio.loadmat(mat_fname)
    img_flt_ref=mat_contents['img_flt']
    img_int_ref=mat_contents['img_int']
    img_flt=img_flt_ref[()]
    img_int=img_int_ref[()]
    
    img_int_sl=np.sum(img_int[:,:,:],axis=-1)
    
    flt_cube_list.append(img_flt)
    int_cube_list.append(img_int)
    int_list.append(img_int_sl)
#%%
cube_siz=np.shape(flt_cube_list[0])
cube_row=cube_siz[0]*3
cube_col=cube_siz[1]*3
cube_page=cube_siz[2]

cube_flt=np.zeros((cube_row,cube_col,cube_page))

cube_int=np.zeros((cube_row,cube_col,cube_page))

# row_start=[0,0,0,512,512,512,1024,1024,1024]
# row_stop=[512,512,512,1024,1024,1024,1536,1536,1536]
# col_start=[0,512,1024,0,512,1024,0,512,1024]
# col_stop=[512,1024,1536,512,1024,1536,512,1024,1536]
row_start=np.array([0,0,0,512,512,512,1024,1024,1024])
row_stop=np.array([512,512,512,1024,1024,1024,1536,1536,1536])
col_start=np.array([0,512,1024,0,512,1024,0,512,1024])
col_stop=np.array([512,1024,1536,512,1024,1536,512,1024,1536])

a=np.array([1,4,7,2,5,8,3,6,9])
ai=np.argsort(a)
row_start=row_start[ai]
row_stop=row_stop[ai]
col_start=col_start[ai]
col_stop=col_stop[ai]

for tile_index in range(onlyfiles_len):
    print('Col start:%s Col stop:%s'%(col_start[tile_index],col_stop[tile_index]))
    print('Row start:%s Row stop:%s'%(row_start[tile_index],row_stop[tile_index]))
    flt_img_tile=flt_cube_list[tile_index]
    int_img_tile=int_cube_list[tile_index]
    # cube_flt[row_start[tile_index]:row_stop[tile_index],col_start[tile_index]:col_stop[tile_index],:]=flt_img_tile
    # cube_int[row_start[tile_index]:row_stop[tile_index],col_start[tile_index]:col_stop[tile_index],:]=int_img_tile
    
    cube_flt[row_start[tile_index]:row_stop[tile_index],col_start[tile_index]:col_stop[tile_index],:]=flt_img_tile
    cube_int[row_start[tile_index]:row_stop[tile_index],col_start[tile_index]:col_stop[tile_index],:]=int_img_tile

# matfile_filename=mypath+'/'+'cube.mat'
# mdic={"cube":cube}
# sio.savemat(matfile_filename, mdic)
#%%
cube_int=ndi.rotate(cube_int,90)
cube_flt=ndi.rotate(cube_flt, 90)

for page in range(cube_page):
    cube_int_slice=np.fliplr(cube_int[:,:,page])
    SNR=np.sqrt(np.mean(cube_int_slice))
    cube_int_slice[cube_int_slice<SNR]=0
    cube_int_slice[cube_int_slice<0]=0
    cube_int[:,:,page]=cube_int_slice
    
    cube_flt_slice=np.fliplr(cube_flt[:,:,page])
    cube_flt_slice[cube_int_slice<SNR]=0
    cube_flt_slice[cube_int_slice<0]=0
    cube_flt_slice[cube_flt_slice<0]=0
    cube_flt_slice[cube_flt_slice>5]=5
    cube_flt[:,:,page]=cube_flt_slice
    
#%%
# SNR=np.sqrt(np.mean(cube_int))
# cube_int[cube_int<SNR]=0
# cube_flt[cube_int<SNR]=0
#%%
plt.figure(1)
plt.subplot(121)
plt.imshow(cube_int[:,:,3],cmap='gray')
plt.colorbar()
plt.subplot(122)
plt.imshow(cube_flt[:,:,3],cmap='gray')
plt.colorbar()
plt.show()