#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 22:54:41 2023

@author: Arun PDRA, THT
"""

#%%
# import os
# import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

import h5py
from scipy import ndimage as ndi
from scipy.io import loadmat, savemat
#%%



# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-4_Col-1_20230214/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-6_Col-10_20230223/Mat_output'


# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/Test_Output/Tiling/Row-1_Col-9_20230222'
base_dir_flt='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-9_20230222/FLT_IMG_DIR'

base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-9_20230222/FLT_IMG_DIR/Stitched'

# mypath_splitted = mypath.replace('/',',')
# mypath_splitted = mypath_splitted.split(',')
# tissue_core_file_name=mypath_splitted[-1]
# save_path ='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/Test_Output'+'/'+tissue_core_file_name


stitch_fiji=cv2.imread(f"{base_dir}/img_t1_z1_c1")
stitch_img_shape=stitch_fiji.shape

tile_size=512

#%%
result_file = open(f"{base_dir}/TileConfiguration.registered.txt", "r")
lines = result_file.readlines()

reg_line_init = "png"

intensity_files = []
lifetime_files = []

positions = {}
shifts = {}
positions_new={}
trunc_slice={}
max_pos_1 = 0
max_pos_2 = 0
min_pos_1 = 10000
min_pos_2 = 10000
for line_num, line in enumerate(lines):
    # if reg_line_init in line:
    #     continue

    # print(f"{line_num}==={line}")
    temp = line.split(";")
    file_name = temp[0].split(":")[-1].strip()
    
    # print(file_name)
    if reg_line_init in file_name:
        # print(file_name)
        
        tile_index = int(file_name.split("_")[-1].split(".")[0])-1
        # tile_index was changed for Py based FLIM reconstruction
        # print(tile_index)
        
        intensity_files.append(file_name)
        
        position = temp[2].split(":")[-1].strip()
        position = position[1:-1]
        x = round(float(position.split(",")[0]))
        y = round(float(position.split(",")[1]))
        positions[tile_index] = [x, y]
        # print('Positions pre shift: %s'%positions[tile_index])
        if max_pos_1 < x:
            max_pos_1 = x
        if max_pos_2 < y:
            max_pos_2 = y
        if min_pos_1 > x:
            min_pos_1 = x
        if min_pos_2 > y:
            min_pos_2 = y

 # This is where the block shifting should be focused on.
        shift_x = 0
        shift_y = 0
        if min_pos_1 < 0:
            shift_x = 0 - min_pos_1
        if min_pos_2 < 0:
            shift_y = 0 - min_pos_2
            
        print('Shift x , y: %s %s'%(shift_x,shift_y))
        shifts[tile_index] = [shift_x, shift_y]

for key, value in positions.items():
    print('Positions pre shift: %s'%positions[key])
    shift_value=shifts[key]
    positions_new[key] = [value[0] + shift_value[0], value[1] + shift_value[1]]
    print('Positions pos shift: %s'%positions_new[key])
    trunc_slice[key]=[shift_value[0],shift_value[1]]
    
no_of_chs=20
# Axis must be swapped from Tile configuration to Python indexing
stitch_intensity_arr = np.zeros([stitch_img_shape[1], stitch_img_shape[0]], dtype=np.uint16)
stitch_intensity = np.zeros([stitch_img_shape[1], stitch_img_shape[0]], dtype=np.uint16)
stitch_intensity_cube  = np.zeros([stitch_img_shape[1], stitch_img_shape[0], no_of_chs], dtype=np.uint16)
stitch_flt_cube  = np.zeros([stitch_img_shape[1], stitch_img_shape[0], no_of_chs], dtype=np.uint16)
stitch_intensity_cube_f  = np.zeros([stitch_img_shape[0], stitch_img_shape[1], no_of_chs], dtype=np.uint16)
stitch_flt_cube_f  = np.zeros([stitch_img_shape[0], stitch_img_shape[1], no_of_chs], dtype=np.uint16)

#%%

# Work on the file order to stitch
img_file_order=np.array([1,4,7,2,5,8,3,6,9])
# mat_file_order=np.array([1,2,3,4,5,6,7,8,9])

# img_file_order=np.array([1,2,3,4,5,6,7,8,9])
mat_file_order=np.array([1,4,7,2,5,8,3,6,9])


for t_i, p in positions_new.items():    
    
    num=int(t_i)+1
    print('Tile %s'%num)
    
    # print('%s %s'%(t_i,p))
    # print(f"{base_dir}/{t_i}.tiff")
    
    img_file_order_index_str=str(img_file_order[int(t_i)])
    intensity1 = cv2.imread(f"{base_dir}/{img_file_order_index_str}.png",-1)
    p=np.array(p)

    print('S P %s'%p)
    
    shift_value=shifts[t_i]   
    p1=p+tile_size-shift_value
    print('E P %s'%p1)

    intensity_blk=intensity1[shift_value[0]:tile_size,shift_value[1]:tile_size]
    intensity_blk_shape=np.array(intensity_blk.shape)
    print('Img Block size: %s'%(intensity_blk_shape))
    
    stitch_intensity_arr[p[0]:p1[0], p[1]:p1[1]] = intensity_blk
    
    mat_file_order_index_str=str(mat_file_order[int(t_i)]-1)
    mat_contents=loadmat(f"{base_dir_flt}/{mat_file_order_index_str}.mat")
    # mat_contents=h5py.File(f"{base_dir}/{t_i}.mat",'r+')

    
    lifetimeAlphaData1_ref=mat_contents['img_int']
    lifetimeAlphaData1=lifetimeAlphaData1_ref[()]
    
    intensity_blk1_cube=lifetimeAlphaData1[shift_value[0]:tile_size,shift_value[1]:tile_size,:]
    intensity_blk1_cube_shape=np.array(intensity_blk1_cube.shape)
    print('Mat Cube size: %s'%(intensity_blk1_cube_shape))
    
    stitch_intensity_cube[p[0]:p1[0], p[1]:p1[1],:] = intensity_blk1_cube

    lifetimeImageData1_ref=mat_contents['img_flt']
    lifetimeImageData1=lifetimeImageData1_ref[()]
    lifetimeImageData1[lifetimeImageData1>10]=10# Crude SNR
    
    flt_blk1_cube=lifetimeImageData1[shift_value[0]:tile_size,shift_value[1]:tile_size,:]
    flt_blk1_cube_shape=np.array(flt_blk1_cube.shape)
    print('Mat Cube size: %s'%(flt_blk1_cube_shape))
    
    stitch_flt_cube[p[0]:p1[0], p[1]:p1[1],:] = flt_blk1_cube
    
    
    intensity_blk1=np.sum(intensity_blk1_cube,axis=-1)
    intensity_blk1_shape=np.array(intensity_blk1.shape)
    print('Mat Block size: %s'%(intensity_blk1_shape))
    
    stitch_intensity[p[0]:p1[0], p[1]:p1[1]] = intensity_blk1
    
    print('Tile finished %s'%num)
    plt.figure(2)
    plt.subplot(3,3,num)
    plt.imshow(intensity1,cmap='gray')
    plt.show()
    
#%%


#%%

#%%

stitch_intensity = np.fliplr(cv2.rotate(stitch_intensity, cv2.ROTATE_90_CLOCKWISE))
stitch_intensity_cube_f  = np.zeros([stitch_img_shape[0], stitch_img_shape[1], no_of_chs], dtype=np.uint16)
stitch_flt_cube_f  = np.zeros([stitch_img_shape[0], stitch_img_shape[1], no_of_chs], dtype=np.uint16)

for page in range(no_of_chs):
    stitch_intensity_cube_f[:,:,page]=np.fliplr(cv2.rotate(stitch_intensity_cube[:,:,page],cv2.ROTATE_90_CLOCKWISE))
    stitch_flt_cube_f[:,:,page]=np.fliplr(cv2.rotate(stitch_flt_cube[:,:,page],cv2.ROTATE_90_CLOCKWISE))
    
#%%
plt.figure(1)
plt.tight_layout()
plt.subplot(1,2,2)
plt.imshow(stitch_intensity, cmap="gray")
plt.title('from mat array')
plt.subplot(1,2,1)
plt.imshow(stitch_intensity_arr, cmap="gray")
plt.title('from img')


page=2
plt.figure(3)
plt.subplot(1,2,1)
plt.imshow(stitch_intensity_cube[:,:,page],cmap='gray')
plt.title('pre rotation')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(stitch_flt_cube[:,:,page],cmap='gray')
plt.colorbar()
plt.show()

plt.figure(4)
plt.subplot(1,2,1)
plt.imshow(stitch_intensity_cube_f[:,:,page],cmap='gray')
plt.title('post rotation')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(stitch_flt_cube_f[:,:,page],cmap='gray')
plt.colorbar()
#%%
mdic={'stitch_intensity':stitch_intensity,'stitch_intensity_cube':stitch_intensity_cube_f,'stitch_flt_cube':stitch_flt_cube_f}
savemat(f"{base_dir}/core_stitched.mat", mdic)