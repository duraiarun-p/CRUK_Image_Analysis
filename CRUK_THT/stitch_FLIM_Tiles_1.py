#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:48:14 2023

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
from scipy.io import savemat
#%%



base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218/Mat_output2'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-4_Col-1_20230214/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-6_Col-10_20230223/Mat_output'


stitch_fiji=cv2.imread(f"{base_dir}/img_t1_z1_c1")
stitch_img_shape=stitch_fiji.shape

tile_size=512



result_file = open(f"{base_dir}/TileConfiguration.registered.txt", "r")
lines = result_file.readlines()

reg_line_init = "tiff"

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
        
        tile_index = file_name.split(".")[0]
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
#     # positions[key] = [value[0], value[1]]
#     sp = np.array( [value[0], value[1]])
#     pos_ind=np.argwhere(sp<0)
#     # sp[pos_ind]=0
#     positions_new[key]=[sp[0] + shift_x, sp[1] + shift_y]
        
        
   
    
no_of_chs=310
# Axis must be swapped from Tile configuration to Python indexing
stitch_intensity_arr = np.zeros([stitch_img_shape[1], stitch_img_shape[0]], dtype=np.uint16)
stitch_intensity = np.zeros([stitch_img_shape[1], stitch_img_shape[0]], dtype=np.uint16)
stitch_intensity_cube  = np.zeros([stitch_img_shape[1], stitch_img_shape[0], no_of_chs], dtype=np.uint16)
stitch_flt_cube  = np.zeros([stitch_img_shape[1], stitch_img_shape[0], no_of_chs], dtype=np.uint16)
stitch_intensity_cube_f  = np.zeros([stitch_img_shape[0], stitch_img_shape[1], no_of_chs], dtype=np.uint16)
stitch_flt_cube_f  = np.zeros([stitch_img_shape[0], stitch_img_shape[1], no_of_chs], dtype=np.uint16)

# stitch_intensity_arr = np.zeros([stitch_img_shape[0] + shift_x, stitch_img_shape[1] + shift_y], dtype=np.uint16)
# stitch_intensity = np.zeros([stitch_img_shape[0] + shift_x, stitch_img_shape[1] + shift_y], dtype=np.uint16)
# stitch_intensity_cube  = np.zeros([stitch_img_shape[0] + shift_x, stitch_img_shape[1] + shift_y, no_of_chs], dtype=np.uint16)

# stitch_intensity_arr = np.zeros([stitch_img_shape[1], stitch_img_shape[0]], dtype=np.uint16)
# stitch_intensity = np.zeros([stitch_img_shape[1], stitch_img_shape[0]], dtype=np.uint16)
# stitch_intensity_cube  = np.zeros([stitch_img_shape[1], stitch_img_shape[0], no_of_chs], dtype=np.uint16)


#%%

img_file_order=np.array([1,4,7,2,5,8,3,6,9])
mat_file_order=np.array([1,2,3,4,5,6,7,8,9])

# for t_i, p in positions.items():
for t_i, p in positions_new.items():    
    
    num=int(t_i)
    print('Tile %s'%num)
    
    # print('%s %s'%(t_i,p))
    # print(f"{base_dir}/{t_i}.tiff")
    img_file_order_index_str=str(img_file_order[int(t_i)-1])
    intensity1 = cv2.imread(f"{base_dir}/{img_file_order_index_str}.tiff",-1)
    
    # mask = (intensity>0).astype(int)
    # if np.sum(mask) < 0.05*intensity.shape[0]*intensity.shape[1]:
    #     continue
    
    # print('%s'%p)
    p=np.array(p)
    # p1=p
    # p1=np.zeros_like(p)
    # p1[0]=p[0]+shifts[t_i][0]
    # p1[1]=p[1]+shifts[t_i][1]
    print('S P %s'%p)
    
    # p_siz=p1-p
    # print(p_siz)
    # print(intensity1.shape)
    # print(p,p1)
    # pos_ind=np.squeeze(np.argwhere(p<0))
    # pos_ind=(np.argwhere(p<0))
    # if len(pos_ind)==0:
    #     print('No neg')
    # else:
    #     print(pos_ind)
    #     print(len(pos_ind))
    # print(pos_ind)
    
    # if len(pos_ind)>0:
        
    
    # stitch_intensity[p[1]:p[1]+tile_size, p[0]:p[0]+tile_size] = intensity1
    # stitch_intensity_arr[p[1]:p[1]+tile_size, p[0]:p[0]+tile_size] = intensity1
    
    shift_value=shifts[t_i]   
    p1=p+tile_size-shift_value
    print('E P %s'%p1)
    # stitch_intensity_arr[p[0]:p[0]+tile_size-shift_value[0], p[1]:p[1]+tile_size-shift_value[1]] = intensity1[shift_value[0]:,shift_value[1]:]
    intensity_blk=intensity1[shift_value[0]:tile_size,shift_value[1]:tile_size]
    intensity_blk_shape=np.array(intensity_blk.shape)
    print('Img Block size: %s'%(intensity_blk_shape))
    
    stitch_intensity_arr[p[0]:p1[0], p[1]:p1[1]] = intensity_blk
    # stitch_intensity_arr[p[0]:p1[0], p[1]:p1[1]] = (stitch_intensity_arr[p[0]:p1[0], p[1]:p1[1]]+intensity_blk)*0.5
    # img_stack=np.stack([stitch_intensity_arr[p[0]:p1[0], p[1]:p1[1]],intensity_blk])
    # intensity_blk_med=np.median(img_stack,axis=0)
    # stitch_intensity_arr[p[0]:p1[0], p[1]:p1[1]] = intensity_blk_med
    
    
    mat_file_order_index_str=str(mat_file_order[int(t_i)-1])
    mat_contents=h5py.File(f"{base_dir}/{mat_file_order_index_str}.mat",'r+')
    # mat_contents=h5py.File(f"{base_dir}/{t_i}.mat",'r+')
    
    allIntensityImages1_ref=mat_contents['allIntensityImages1']
    allIntensityImages1=allIntensityImages1_ref[()]    
    
    intensity_blk1=allIntensityImages1[shift_value[0]:tile_size,shift_value[1]:tile_size]
    intensity_blk1_shape=np.array(intensity_blk1.shape)
    print('Mat Block size: %s'%(intensity_blk1_shape))
    
    stitch_intensity[p[0]:p1[0], p[1]:p1[1]] = intensity_blk1
    # stitch_intensity[p[0]:p1[0], p[1]:p1[1]] = (stitch_intensity[p[0]:p1[0], p[1]:p1[1]]+intensity_blk1)*0.5
    
    # img_stack_int=np.stack([stitch_intensity[p[0]:p1[0], p[1]:p1[1]],intensity_blk])
    # intensity_blk_med_1=np.median(img_stack_int,axis=0)
    # stitch_intensity[p[0]:p1[0], p[1]:p1[1]] = intensity_blk_med_1
    
    lifetimeAlphaData1_ref=mat_contents['lifetimeAlphaData1']
    lifetimeAlphaData1=lifetimeAlphaData1_ref[()]
    
    intensity_blk1_cube=lifetimeAlphaData1[shift_value[0]:tile_size,shift_value[1]:tile_size,:]
    intensity_blk1_cube_shape=np.array(intensity_blk1_cube.shape)
    print('Mat Cube size: %s'%(intensity_blk1_cube_shape))
    
    stitch_intensity_cube[p[0]:p1[0], p[1]:p1[1],:] = intensity_blk1_cube
    # stitch_intensity_cube[p[0]:p1[0], p[1]:p1[1],:] = (stitch_intensity_cube[p[0]:p1[0], p[1]:p1[1],:]+intensity_blk1_cube)*0.5

    

    lifetimeImageData1_ref=mat_contents['lifetimeImageData1']
    lifetimeImageData1=lifetimeImageData1_ref[()]
    
    flt_blk1_cube=lifetimeImageData1[shift_value[0]:tile_size,shift_value[1]:tile_size,:]
    flt_blk1_cube_shape=np.array(flt_blk1_cube.shape)
    print('Mat Cube size: %s'%(flt_blk1_cube_shape))
    
    stitch_flt_cube[p[0]:p1[0], p[1]:p1[1],:] = flt_blk1_cube
    
    
    # stitch_intensity_cube[p[1]:p[1]+tile_size, p[0]:p[0]+tile_size, :] = lifetimeAlphaData1
    
    # stitch_intensity[p[1]:p[1]+tile_size, p[0]:p[0]+tile_size] = allIntensityImages1
    # stitch_intensity[p[1]:p[1]+tile_size, p[0]:p[0]+tile_size] = allIntensityImages1[shift_ind2:,-shift_ind1:]
    
    # stitch_intensity[p[0]:p[0]+tile_size, p[1]:p[1]+tile_size] = allIntensityImages1
    
    # stitch_intensity[p1[0]:p1[0]+tile_size, p1[1]:p1[1]+tile_size] = allIntensityImages1
    # stitch_intensity[p1[0]:p1[0]+512, p1[1]:p1[1]+512] = allIntensityImages1
    
    # stitch_intensity[p[0]:p[0]+tile_size, p[1]:p[1]+tile_size] = allIntensityImages1
    # stitch_intensity_cube[p[0]:p[0]+tile_size, p[1]:p[1]+tile_size, :] = lifetimeAlphaData1
    
    # stitch_intensity[p[0]:p[0]+tile_size, p[1]:p[1]+tile_size] = np.flipud(ndi.rotate(allIntensityImages1,90))
    # stitch_intensity_cube[p[0]:p[0]+tile_size, p[1]:p[1]+tile_size, :] = np.flipud(ndi.rotate(lifetimeAlphaData1,90))
    
    # num=int(t_i)
    print('Tile finished %s'%num)
    plt.figure(2)
    plt.subplot(3,3,num)
    plt.imshow(intensity1,cmap='gray')
    plt.show()

    
#%%

# stitch_intensity=np.flipud(ndi.rotate(stitch_intensity,90))
# stitch_intensity_cube=ndi.rotate(stitch_intensity_cube, 90)

stitch_intensity = np.flipud(cv2.rotate(stitch_intensity, cv2.ROTATE_90_COUNTERCLOCKWISE))

for page in range(no_of_chs):
    stitch_intensity_cube_f[:,:,page]=np.flipud(cv2.rotate(stitch_intensity_cube[:,:,page],cv2.ROTATE_90_COUNTERCLOCKWISE))
    stitch_flt_cube_f[:,:,page]=np.flipud(cv2.rotate(stitch_flt_cube[:,:,page],cv2.ROTATE_90_COUNTERCLOCKWISE))
    
#%%
mdic={'stitch_intensity':stitch_intensity,'stitch_intensity_cube':stitch_intensity_cube_f,'stitch_flt_cube':stitch_flt_cube_f}
# savemat(f"{base_dir}/core_stitched.mat", mdic)

#%%

plt.figure(1)
plt.tight_layout()
plt.subplot(1,2,2)
plt.imshow(stitch_intensity, cmap="gray")
plt.title('from mat array')
plt.subplot(1,2,1)
plt.imshow(stitch_intensity_arr, cmap="gray")
plt.title('from img')

#%%
page=100
plt.figure(3)
plt.subplot(1,2,1)
plt.imshow(stitch_intensity_cube[:,:,page],cmap='gray')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(stitch_flt_cube[:,:,page],cmap='gray')
plt.colorbar()
plt.show()
plt.figure(4)
plt.subplot(1,2,1)
plt.imshow(stitch_intensity_cube_f[:,:,page],cmap='gray')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(stitch_flt_cube_f[:,:,page],cmap='gray')
plt.colorbar()
plt.show()
#%%

# plt.figure(4)
# plt.imshow(stitch_fiji,cmap='gray')
# plt.show()
#%%
plt.figure(5)
plt.subplot(1,2,1)
plt.imshow(stitch_fiji)
plt.title('Fiji Output')
plt.subplot(1,2,2)
plt.imshow(stitch_intensity,cmap='gray')
plt.title('Mat stitched')
plt.show()
plt.savefig(f"{base_dir}/stitched_compared.png")