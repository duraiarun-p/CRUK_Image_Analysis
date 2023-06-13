#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:09:38 2023

@author: Arun PDRA, THT
"""

import h5py

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
import imageio
# from xtiff import to_tiff

#%%


def comp_tiling(mypath):
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
        img_flt[img_flt>5]=5 # Upper threshold
        
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
    
    tma_int=np.zeros((cube_row,cube_col))
    
    
    
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
        # print('Col start:%s Col stop:%s'%(col_start[tile_index],col_stop[tile_index]))
        # print('Row start:%s Row stop:%s'%(row_start[tile_index],row_stop[tile_index]))
        
        flt_img_tile=flt_cube_list[tile_index]
        int_img_tile=int_cube_list[tile_index]
        int_list_sl=int_list[tile_index]
        
        # flt_img_tile=ndi.median_filter(flt_img_tile,size=3)
        # int_img_tile=ndi.median_filter(int_img_tile,size=3)
        # int_list_sl=ndi.median_filter(int_list_sl,size=3)
        
        
        cube_flt[row_start[tile_index]:row_stop[tile_index],col_start[tile_index]:col_stop[tile_index],:]=flt_img_tile
        cube_int[row_start[tile_index]:row_stop[tile_index],col_start[tile_index]:col_stop[tile_index],:]=int_img_tile
        tma_int[row_start[tile_index]:row_stop[tile_index],col_start[tile_index]:col_stop[tile_index]]=int_list_sl
        
        
    
        
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
    
    tma_int=np.fliplr(ndi.rotate(tma_int, 90))
    
    #%%
    
    #%%
    # row_start=row_start
    # row_stop=row_stop
    # col_start=col_start
    # col_stop=col_stop[ai]
    overlap_size=12
    tma_int_f=np.zeros((cube_row-(6*overlap_size),cube_col-(6*overlap_size)))
    cube_flt_f=np.zeros((cube_row-(6*overlap_size),cube_col-(6*overlap_size),cube_page))
    cube_int_f=np.zeros((cube_row-(6*overlap_size),cube_col-(6*overlap_size),cube_page))
    
    # for tile_index in range(onlyfiles_len):
    #     row_x=max(0,row_start[tile_index]-overlap_size)
    #     # row_x=min(row_start[tile_index],row_start[tile_index]-overlap_size)
    #     # row_X=min(row_stop[tile_index],row_stop[tile_index]-overlap_size)
    #     row_X=row_stop[tile_index]-overlap_size
    #     col_x=max(0,col_start[tile_index]-overlap_size)
    #     # col_x=min(col_start[tile_index],col_start[tile_index]-overlap_size)
    #     # col_X=min(col_stop[tile_index],col_stop[tile_index]-overlap_size)
    #     col_X=col_stop[tile_index]-overlap_size
        
    #     print('Col start:%s Col stop:%s'%(col_x,col_X))
    #     print('Row start:%s Row stop:%s'%(row_x,row_X))
        
    #     print('Col start:%s Col stop:%s'%(col_start[tile_index],col_stop[tile_index]))
    #     print('Row start:%s Row stop:%s'%(row_start[tile_index],row_stop[tile_index]))
        
    
    #     # tma_int_f[row_x:row_X,col_x:col_X]=tma_int[row_start[tile_index]:row_stop[tile_index],col_start[tile_index]:col_stop[tile_index]]
        
    #     tma_int_f[row_x:row_X,col_x:col_X]=tma_int[row_start[tile_index]:row_stop[tile_index]-overlap_size,col_start[tile_index]:col_stop[tile_index]-overlap_size]
    
    
    
    
    # tma_int_f[0:488,0:488]=tma_int[0:488,0:488]
    # tma_int_f[0:488,488:976]=tma_int[0:488,512:1000]
    # tma_int_f[0:488,976:1464]=tma_int[0:488,1024:1512]
    
    #         # tma_int_f[488:976,0:488]=tma_int[512:1000,0:488]
    #         # tma_int_f[488:976,488:976]=tma_int[512:1000,512:1000]
    #         # tma_int_f[488:976,976:1464]=tma_int[512:1000,1024:1512]
    
    # tma_int_f[488:976,0:488]=tma_int[536:1024,0:488]
    # tma_int_f[488:976,488:976]=tma_int[536:1024,512:1000]
    # tma_int_f[488:976,976:1464]=tma_int[536:1024,1024:1512]
    
    #             # tma_int_f[976:1464,0:488]=tma_int[1048:1536,0:488]
    #             # tma_int_f[976:1464,488:976]=tma_int[1048:1536,512:1000]
    #             # tma_int_f[976:1464,976:1464]=tma_int[1048:1536,1024:1512]
    
    # tma_int_f[976:1440,0:488]=tma_int[1072:1536,0:488]
    # tma_int_f[976:1440,488:976]=tma_int[1072:1536,512:1000]
    # tma_int_f[976:1440,976:1464]=tma_int[1072:1536,1024:1512]
    
    
    # row_step=cube_siz[0]-(2*overlap_size)
    # col_step=cube_siz[1]-(2*overlap_size)
    
    
    
    def img_tiling(tma_intx,tma_int_fx):
        
        if len(tma_intx.shape)>2:
            tma_int_fx[0:488,0:488,:]=tma_intx[0:488,0:488,:]
            tma_int_fx[0:488,488:976,:]=tma_intx[0:488,512:1000,:]
            tma_int_fx[0:488,976:1464,:]=tma_intx[0:488,1024:1512,:]
    
            tma_int_fx[488:976,0:488,:]=tma_intx[536:1024,0:488,:]
            tma_int_fx[488:976,488:976,:]=tma_intx[536:1024,512:1000,:]
            tma_int_fx[488:976,976:1464,:]=tma_intx[536:1024,1024:1512,:]
    
            tma_int_fx[976:1440,0:488,:]=tma_intx[1072:1536,0:488,:]
            tma_int_fx[976:1440,488:976,:]=tma_intx[1072:1536,512:1000,:]
            tma_int_fx[976:1440,976:1464,:]=tma_intx[1072:1536,1024:1512,:]
        else:
            tma_int_fx[0:488,0:488]=tma_intx[0:488,0:488]
            tma_int_fx[0:488,488:976]=tma_intx[0:488,512:1000]
            tma_int_fx[0:488,976:1464]=tma_intx[0:488,1024:1512]
    
            tma_int_fx[488:976,0:488]=tma_intx[536:1024,0:488]
            tma_int_fx[488:976,488:976]=tma_intx[536:1024,512:1000]
            tma_int_fx[488:976,976:1464]=tma_intx[536:1024,1024:1512]
    
            tma_int_fx[976:1440,0:488]=tma_intx[1072:1536,0:488]
            tma_int_fx[976:1440,488:976]=tma_intx[1072:1536,512:1000]
            tma_int_fx[976:1440,976:1464]=tma_intx[1072:1536,1024:1512]
        return tma_int_fx
    
    def img_tile_remove(tma_int_fx):
        tma_int_fx[:,487]=(tma_int_fx[:,488]+tma_int_fx[:,489])*0.5
        tma_int_fx[:,975]=(tma_int_fx[:,974]+tma_int_fx[:,976])*0.5
        tma_int_fx[487,:]=(tma_int_fx[488,:]+tma_int_fx[489,:])*0.5
        tma_int_fx[975,:]=(tma_int_fx[974,:]+tma_int_fx[976,:])*0.5
        
        # tma_fx_arr=
        return tma_int_fx
    
    #%%
    
    
    tma_int_f=img_tiling(tma_int,tma_int_f)
    cube_flt_f=img_tiling(cube_flt,cube_flt_f)
    cube_int_f=img_tiling(cube_int,cube_int_f)
    
    return tma_int_f,cube_flt_f,cube_int_f

#%%