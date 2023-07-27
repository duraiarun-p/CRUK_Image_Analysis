#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:31:29 2023

@author: Arun PDRA, THT
"""

import os
import sys
import csv
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatch
from scipy.io import savemat,loadmat
import numpy as np
#%%
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output'
base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218/Mat_output2'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-4_Col-1_20230214/Mat_output2'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-6_Col-10_20230223/Mat_output2'
#%% File Check

file_check='classification_QuPath.txt'

path_to_class_file=base_dir+'/'+file_check
if os.path.exists(path_to_class_file)==False:
    sys.exit(f"Execution was stopped due to cell {file_check} file was not found error")

#%% Validating bounding box by overlaying them on original figure

file_extension_type = ('.tif',) # , '.exe', 'jpg', '...')
for hist_file in os.listdir(base_dir):
    if hist_file.endswith(file_extension_type) and hist_file.startswith('R'):
        print("Found a file {}".format(hist_file)) 
        hist_img=cv2.imread(f"{base_dir}/{hist_file}")    
        
file_extension_type = ('.tiff',) # , '.exe', 'jpg', '...')
for hist_file in os.listdir(base_dir):
    if hist_file.endswith(file_extension_type) and hist_file.startswith('h'):
        print("Found a file {}".format(hist_file)) 
        hist_img_R=cv2.imread(f"{base_dir}/{hist_file}")
    

#%% Extracting information from QuPATH classification output
with open(path_to_class_file) as class_file:
    lines = [line.rstrip('\n') for line in class_file]
# using tab as a separator
items = [item.split('\t') for item in lines]

del lines

column_names=items[0] # Extracting table column names

del items[0]
# Convert numeric values into float datatype
for item_idx in range(len(items)):
    item=items[item_idx]
    for ele_idx in range(len(item)):
        ele=item[ele_idx]
        try:
            item[ele_idx]=float(ele)
        except ValueError:
            item[ele_idx]=ele
    items[item_idx]=item

#%% Extracting cell centroid from the QuPath output file

# # Index of column in items list in order

cell_centroid_x_ind=column_names.index("Centroid X px")
cell_centroid_y_ind=column_names.index("Centroid Y px")
cell_class=column_names.index("Class")
cell_roi=column_names.index("ROI")
cell_area=column_names.index("Cell: Area")
cell_perimeter=column_names.index("Cell: Perimeter")
cell_caliper_max=column_names.index("Cell: Max caliper")
cell_caliper_min=column_names.index("Cell: Min caliper")

box_space=0.35
cell_items=[]
for item_idx in range(len(items)):
    # Bounding box x co-ordinates
    bound_x=[items[item_idx][cell_centroid_x_ind]+(items[item_idx][cell_caliper_max])*box_space,
              items[item_idx][cell_centroid_x_ind]-(items[item_idx][cell_caliper_max])*box_space,
              items[item_idx][cell_centroid_x_ind]+(items[item_idx][cell_caliper_min])*box_space,
              items[item_idx][cell_centroid_x_ind]-(items[item_idx][cell_caliper_min])*box_space
                ]
    bound_x=[min(bound_x),max(bound_x)]
    
    bound_y=[items[item_idx][cell_centroid_y_ind]+(items[item_idx][cell_caliper_max])*box_space,
              items[item_idx][cell_centroid_y_ind]-(items[item_idx][cell_caliper_max])*box_space,
              items[item_idx][cell_centroid_y_ind]+(items[item_idx][cell_caliper_min])*box_space,
              items[item_idx][cell_centroid_y_ind]-(items[item_idx][cell_caliper_min])*box_space
                ]
    bound_y=[min(bound_y),max(bound_y)]
    
    bound_area=abs(bound_x[1]-bound_x[0])*abs(bound_y[1]-bound_y[0])
    
    cell_item=[items[item_idx][cell_centroid_x_ind],items[item_idx][cell_centroid_y_ind],
                items[item_idx][cell_class],items[item_idx][cell_roi],items[item_idx][cell_area],items[item_idx][cell_perimeter],
                items[item_idx][cell_caliper_max],items[item_idx][cell_caliper_min],bound_x,bound_y,bound_area]
    
    cell_items.append(cell_item)

#%% Test Points

cell_plot_index=np.arange(len(cell_items),step=1000)

fig=plt.figure(2),
plt.imshow(hist_img),
# point_ox=[]
for cell_plt_ind in range(len(cell_plot_index)):
    item_idx=cell_plot_index[cell_plt_ind]
    # print(item_idx)
    cell_item=cell_items[item_idx] 
    # point_ox.append(cell_item)
    
    plt.plot(cell_item[0],cell_item[1],marker='o', color="r")
    xy=(cell_item[-3][0],cell_item[-2][0]) # Choose min of bound x and y
    x_width=abs(cell_item[-3][0]-cell_item[-3][1])
    y_width=abs(cell_item[-2][0]-cell_item[-2][1])
    plt.plot(xy[0],xy[1],marker='o', color="k")
    rect=pltpatch.Rectangle(xy, x_width, y_width,linewidth=1, edgecolor='k', facecolor='none')
    plt.gca().add_patch(rect)
#%% Place for validation of Geometric Transform
tx_siz=hist_img_R.shape # Registered Image Shape
ox_siz=hist_img.shape # Original Image Shape
# tx_f_siz=np.round((np.array(ox_siz)/np.array(tx_siz)))
tx_f_siz=np.array(ox_siz)/np.array(tx_siz)
tx_f_siz1=((np.array(tx_siz)/np.array(ox_siz)))# Proper scale factor for this implemetnation
ox_mid=np.array(ox_siz[:2])/2-1# Mid point of original image

tx_mid=np.array(tx_siz[:2])/2-1# Mid point of registered image

tx_mid1=np.zeros_like(ox_mid)
tx_mid1[0]=(ox_mid[0]-ox_mid[0])*tx_f_siz1[0]+tx_mid[0]
tx_mid1[1]=(ox_mid[1]-ox_mid[1])*tx_f_siz1[1]+tx_mid[1]

hist_img_R1=cv2.resize(hist_img,(tx_siz[0],tx_siz[1]), interpolation= cv2.INTER_NEAREST) # Resizing original image for 1st stage transform

# Take note of the transformation matrix indices 
# The matrix order should not be changed otherwise you will mess up the pixel mapping
# Resizing is Scaling Transformation and below is the Scaling transformation matrix
# T_S=np.array([[tx_f_siz1[0], 0, 0], 
#               [0, tx_f_siz1[1], 0], 
#               [0, 0, 1]])

T_S=np.array([[tx_f_siz1[1], 0, 0], 
              [0, tx_f_siz1[0], 0], 
              [0, 0, 1]])

# trans = T_S[0:2, :]
# inv_t = np.linalg.inv(T_S)
# inv_trans = inv_t[0:2, :]

# h, w = src_im.shape[:2]
# src_pts = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
# src_pts = np.float32([[0, 0], [ox_siz[1]-1, 0], [0, ox_siz[0]-1], [ox_siz[0]-1, ox_siz[1]-1]]) # https://stackoverflow.com/questions/44378098/trouble-getting-cv-transform-to-work (see comment).
# dst_pts = cv2.transform(np.array([src_pts]), trans)[0]

# min_x, max_x = np.min(dst_pts[:, 0]), np.max(dst_pts[:, 0])
# min_y, max_y = np.min(dst_pts[:, 1]), np.max(dst_pts[:, 1])

# # Destination matrix width and height
# dst_w = int(max_x - min_x + 1) # 895
# dst_h = int(max_y - min_y + 1) # 384

# dst_center = np.float32([[(dst_w-1.0)/2, (dst_h-1.0)/2]])
# src_projected_center = cv2.transform(np.array([dst_center]), inv_trans)[0]

# # Compute the translation of the center - assume source center goes to destination center
# translation = src_projected_center - np.float32([[(ox_siz[1]-1.0)/2, (ox_siz[0]-1.0)/2]])

# # Place the translation in the third column of trans
# trans[:, 2] = translation

T_S1=np.array([[tx_f_siz1[0], 0, 0], 
              [0, tx_f_siz1[1], 0]
              ])
# T_S1=np.linalg.inv(T_S);

#%% Key subroutine for coordinate transformation 
trans = T_S[0:2, :]
t_inv = np.linalg.inv(T_S)
trans_inv = t_inv[0:2, :]


# src_pts = np.float32([[0, 0], [ox_siz[1]-1, 0], [0, ox_siz[0]-1], [ox_siz[1]-1, ox_siz[0]-1]])
src_pts = np.float32([[0, 0], [ox_siz[0]-1, 0], [0, ox_siz[1]-1], [ox_siz[0]-1, ox_siz[1]-1]])
dst_pts = cv2.transform(np.array([src_pts]), trans_inv)[0]


# min_x, max_x = np.min(dst_pts[:, 0]), np.max(dst_pts[:, 0])
# min_y, max_y = np.min(dst_pts[:, 1]), np.max(dst_pts[:, 1])

min_x, max_x = tx_mid[0] - ox_mid[0] , tx_mid[0] + ox_mid[0] 
min_y, max_y = tx_mid[1] - ox_mid[1] , tx_mid[1] + ox_mid[1] 

# Destination matrix width and height
dst_w = int(max_x - min_x + 1) # 895
dst_h = int(max_y - min_y + 1) # 384

dst_center = np.float32([[(dst_w-1.0)/2, (dst_h-1.0)/2]])


tx_mid_1=(np.reshape(np.array(tx_mid),(1,2)))
ox_mid_1=(np.reshape(np.array(ox_mid),(1,2)))

# tx_mid_1=np.flip(np.reshape(np.array(tx_mid),(1,2)))
# ox_mid_1=np.flip(np.reshape(np.array(ox_mid),(1,2)))

src_projected_center = cv2.transform(np.array([tx_mid_1]), trans_inv)[0]
ox_mid_1=(np.reshape(np.array(ox_mid),(1,2)))
# translation = (src_projected_center - np.float32(ox_mid_1))


translation = (src_projected_center - np.float32(dst_center))

# translation = np.flip(translation)
# translation = (np.float32(ox_mid_1) - src_projected_center)
# translation[0,0]=-66.557
# translation[0,1]=88.2634
# T_S[:2,2]=translation
trans[:, 2] = translation

trans1=tx_mid-(ox_mid/tx_f_siz[:2])


#%% 
def zoom(image, ratio, points, canvas_off_x, canvas_off_y):
    width, height = image.shape[:2]
    new_width, new_height = int(ratio * width), int(ratio * height)
    center_x, center_y = int(new_width / 2), int(new_height / 2)
    radius_x, radius_y = int(width / 2), int(height / 2)
    min_x, max_x = center_x - radius_x, center_x + radius_x
    min_y, max_y = center_y - radius_y, center_y + radius_y
    img_resized = cv2.resize(image, (new_width,new_height), interpolation=cv2.INTER_LINEAR)   
    img_cropped = img_resized[min_y:max_y+1, min_x:max_x+1]
    for point in points:
        x, y = point.get_original_coordinates()
        x -= canvas_off_x
        y -= canvas_off_y
        x = int((x * ratio) - min_x + canvas_off_x)
        y = int((y * ratio) - min_y + canvas_off_y)
        point.set_scaled_coordinates(x, y)
#%%
fig=plt.figure(3),
plt.imshow(hist_img_R1),
point_ox=[]
point_tx=[]
# # plt.plot(cell_item_tx_f[0],cell_item_tx_f[1],marker='o', color="r")
for cell_plt_ind in range(len(cell_plot_index)):
    item_idx=cell_plot_index[cell_plt_ind]
    # cell_item_tx=np.array(cell_items[item_idx][:2],dtype=np.float64)
    cell_item_tx=cell_items[item_idx][:2]
    cell_item_tx.append(1)
    cell_item_tx=np.round(np.array(cell_item_tx,dtype=np.float64))
    # cell_item_tx_f=np.zeros_like(cell_item_tx)
#     # cell_item_tx_f[0]=((cell_item_tx[0])*0.49) + 8.9332
#     # cell_item_tx_f[1]=((cell_item_tx[1])*0.5277) + (-4.8659)
#     # cell_item_tx_f[0]=((cell_item_tx[0]-ox_mid[0])*tx_f_siz1[0]) + tx_mid[0]
#     # cell_item_tx_f[1]=((cell_item_tx[1]-ox_mid[1])*tx_f_siz1[1]) + tx_mid[1]
    
#     cell_item_tx_f[0]=((cell_item_tx[0])*tx_f_siz1[0]) - 1
#     cell_item_tx_f[1]=((cell_item_tx[1])*tx_f_siz1[1]) - 1
    cell_item_tx_f=T_S @ cell_item_tx
    # cell_item_tx_f=cell_item_tx @ T_S1
    # cell_item_tx_f=cv2.transform(cell_item_tx,T_S)
    # cell_item_tx_f = forwardAffineTransform(T_S,cell_item_tx[0],cell_item_tx[1])
    # cell_item_tx_f = np.dot(T_S,cell_item_tx)
    point_ox.append(cell_item_tx)
    point_tx.append(cell_item_tx_f)
    
    plt.plot(cell_item_tx_f[0],cell_item_tx_f[1],marker='o', color="g")
    
#%%
# r_new=[86.2880669419436,
# 958.740150792510,
# 592.100767356166,
# 1219.12893739628,
# 628.016462060134]
# c_new=[492.928231400491,
# 143.320464410200,
# 1010.29133819661,
# 769.231144021850,
# 547.906872177190]
# fig=plt.figure(4),
# plt.imshow(hist_img_R1),
# for pi in range(len(r_new)):
#     plt.plot(r_new[pi],c_new[pi],marker='o', color="g")
